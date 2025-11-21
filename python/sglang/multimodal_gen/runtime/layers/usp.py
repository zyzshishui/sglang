# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import inspect
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from packaging.version import parse
from torch.distributed.tensor.experimental._attention import _cp_options

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

_cp_options.enable_load_balance = False

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
        AttentionImpl,
    )

logger = logging.getLogger(__name__)


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _usp_all_to_all_single(x: torch.Tensor) -> torch.Tensor:
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(
        x, output_split_sizes=None, input_split_sizes=None, group=ulysses_pg
    )
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _usp_input_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style input all-to-all over the head dimension.

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h, s_local, d] -> [b, h // world_size, s_global, d]

    If heads are at dim=2 (input is [b, s_local, h, d]), set head_dim=2, and the
    function returns [b, s_global, h // world_size, d], preserving the original
    head/sequence dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads sharded and sequence gathered.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape
    assert (
        h % world_size == 0
    ), f"h ({h}) must be divisible by world_size ({world_size})"

    # [b, h, s, d] -> [h, b, s, d]
    x_c = x_c.permute(1, 0, 2, 3).contiguous()
    # all-to-all along h
    x_c = _usp_all_to_all_single(x_c)
    # -> [b, h // world, s * world, d]
    x_c = (
        x_c.reshape(world_size, h // world_size, b, -1, d)
        .permute(2, 1, 0, 3, 4)
        .reshape(b, h // world_size, -1, d)
    )

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


def _usp_output_all_to_all(x: torch.Tensor, head_dim: int = 1) -> torch.Tensor:
    """
    Perform Ulysses-style output all-to-all over the head dimension (inverse of input).

    Default layout expects heads at dim=1 and sequence at dim=2:
        [b, h // world_size, s_global, d] -> [b, h, s_local, d]

    If heads are at dim=2 (input is [b, s_global, h // world_size, d]), set head_dim=2,
    and the function returns [b, s_local, h, d], preserving the original head/sequence
    dim ordering.

    Args:
        x: A 4D tensor with layout [b, *, *, d] where '*' are sequence and heads
        head_dim: Which dimension index corresponds to heads (1 or 2)

    Returns:
        Tensor with the same dim order as input, with heads gathered and sequence sharded.
    """
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, f"x must have 4 dimensions, got {x.ndim}"
    assert head_dim in (1, 2), f"head_dim must be 1 or 2, got {head_dim}"
    seq_dim = 1 if head_dim == 2 else 2

    # Bring to canonical [b, h, s, d]
    if head_dim == 1 and seq_dim == 2:
        x_c = x
    else:
        x_c = x.permute(0, head_dim, seq_dim, 3).contiguous()

    b, h, s, d = x_c.shape
    assert (
        s % world_size == 0
    ), f"s ({s}) must be divisible by world_size ({world_size})"

    # [b, h, s, d] -> [s, b, h, d]
    x_c = x_c.permute(2, 0, 1, 3).contiguous()
    x_c = _usp_all_to_all_single(x_c)
    # -> [b, h * world, s // world, d]
    x_c = (
        x_c.reshape(world_size, s // world_size, b, -1, d)
        .permute(2, 0, 3, 1, 4)
        .reshape(b, -1, s // world_size, d)
    )

    if head_dim == 1 and seq_dim == 2:
        return x_c

    # Map back to original ordering, preserving head/seq positions
    new_order = [0, None, None, 3]
    new_order[head_dim] = 1
    new_order[seq_dim] = 2
    return x_c.permute(tuple(new_order)).contiguous()


def ring_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_impl: "AttentionImpl",
    is_causal: bool = False,
    dropout_p: float = 0.0,
):
    """
    Ring Attention implementation.

    This function implements Ring Attention, a strategy for distributed attention
    computation that reduces peak memory usage. It accepts a generic attention
    implementation (`attn_impl`) which is called by the underlying PyTorch
    distributed attention primitive.

    Args:
        query, key, value: The input tensors for attention.
        attn_impl: An instance of an attention implementation backend
                   (e.g., FlashAttentionImpl) whose `forward` method will be
                   used as the computational kernel.
        is_causal: Whether to apply causal masking.
        dropout_p: Dropout probability.
    """
    # torch.distributed.tensor.experimental._attention is not a public API,
    from torch.distributed.tensor.experimental._attention import (
        _templated_ring_attention,
    )

    ring_pg = get_sp_group().ring_group
    assert ring_pg is not None, "Ring process group is not initialized."

    # Ring attention primitives expect tensors in [B, H, S, D] layout.
    # We permute the inputs here.
    query = torch.permute(query, [0, 2, 1, 3]).contiguous()
    key = torch.permute(key, [0, 2, 1, 3]).contiguous()
    value = torch.permute(value, [0, 2, 1, 3]).contiguous()

    # Create an adapter function that matches the signature expected by
    # _templated_ring_attention. The `attn_impl` already has dropout and
    # causal settings configured during its initialization.
    forward_params = inspect.signature(attn_impl.forward).parameters
    supports_return_lse = "return_softmax_lse" in forward_params

    def _compute_softmax_lse(
        q: torch.Tensor, k: torch.Tensor, is_causal_flag: bool, chunk_size: int = 128
    ) -> torch.Tensor:
        # Compute logsumexp over key dimension without materializing the full
        # attention matrix. This avoids OOM when sequence length is large.
        scale = getattr(attn_impl, "softmax_scale", 1.0)
        q_heads = q.transpose(1, 2).to(torch.float32)  # [B, H, S, D]
        k_heads = k.transpose(1, 2).to(torch.float32)  # [B, H_k, S, D]

        if k_heads.shape[1] != q_heads.shape[1]:
            assert (
                q_heads.shape[1] % k_heads.shape[1] == 0
            ), "Q/K heads must be compatible for GQA"
            repeat = q_heads.shape[1] // k_heads.shape[1]
            k_heads = k_heads.repeat_interleave(repeat, dim=1)

        b, h, s_q, _ = q_heads.shape
        lse = torch.full((b, h, s_q), float("-inf"), device=q_heads.device)

        q_pos = (
            torch.arange(s_q, device=q_heads.device).view(1, 1, s_q, 1).to(torch.int64)
        )

        for start in range(0, k_heads.shape[2], chunk_size):
            end = min(start + chunk_size, k_heads.shape[2])
            k_chunk = k_heads[:, :, start:end, :]  # [B, H, chunk, D]
            logits = torch.matmul(
                q_heads, k_chunk.transpose(-1, -2)
            )  # [B, H, S, chunk]
            logits = logits * scale

            if is_causal_flag:
                k_pos = (
                    torch.arange(start, end, device=q_heads.device)
                    .view(1, 1, 1, -1)
                    .to(torch.int64)
                )
                logits = logits.masked_fill(k_pos > q_pos, float("-inf"))

            chunk_lse = torch.logsumexp(logits, dim=-1)
            lse = torch.logaddexp(lse, chunk_lse)

        return lse

    # Note: Please be aware that Attention Backend and Ring Attention may require different QKV tensor shapes.
    # For example, FlashAttention expects the format to be BSHD.
    def attn_callable_adapter(q, k, v, *args, **kwargs):
        # We ignore the dropout_p and is_causal passed by _templated_ring_attention
        # and rely on the pre-configured attn_impl.
        # The `attn_metadata` is not available here, so we pass None.
        # This is a limitation we must accept when using this experimental API.
        q = torch.permute(q, [0, 2, 1, 3])
        k = torch.permute(k, [0, 2, 1, 3])
        v = torch.permute(v, [0, 2, 1, 3])
        is_causal_flag = bool(kwargs.get("is_causal", False))
        rest = []

        if supports_return_lse:
            attn_out = attn_impl.forward(
                q,
                k,
                v,
                attn_metadata=None,
                return_softmax_lse=True,
            )
        else:
            attn_out = attn_impl.forward(
                q,
                k,
                v,
                attn_metadata=None,
            )

        if isinstance(attn_out, tuple):
            output, softmax_lse, *rest = attn_out
        else:
            output, softmax_lse = attn_out, None

        if softmax_lse is None:
            softmax_lse = _compute_softmax_lse(q, k, is_causal_flag)

        output = torch.permute(output, [0, 2, 1, 3])
        return output, softmax_lse, *rest

    # Starting from torch 2.6.0, _templated_ring_attention expects an integer
    # segment_id for the attention function.
    use_segment_id = parse(torch.__version__).release >= parse("2.6.0").release

    # Torch changed the signature of _templated_ring_attention several times.
    # Instead of relying on version numbers, build kwargs based on the actual
    # callable signature to stay compatible with both old (mesh/dropout_p) and
    # new (group/seq_dim) forms.
    signature_params = inspect.signature(_templated_ring_attention).parameters
    attn_kwargs = {
        "op": attn_callable_adapter,
        "is_causal": is_causal,
        "query": query,
        "key": key,
        "value": value,
    }

    if "group" in signature_params:
        attn_kwargs["group"] = ring_pg
    elif "mesh" in signature_params:
        attn_kwargs["mesh"] = ring_pg
    else:
        raise RuntimeError(
            "Unsupported _templated_ring_attention signature: missing group/mesh"
        )

    if "dropout_p" in signature_params:
        attn_kwargs["dropout_p"] = dropout_p

    seq_dim_index = 2  # sequence dimension for [B, H, S, D]
    if "seq_dim" in signature_params:
        attn_kwargs["seq_dim"] = seq_dim_index
    elif use_segment_id:
        # Older builds may still refer to this as segment_id.
        attn_kwargs["segment_id"] = seq_dim_index

    # Fast path: backend returns LSE so we can use ring attention directly.
    if supports_return_lse:
        out, *_ = _templated_ring_attention(**attn_kwargs)
        output = torch.permute(out, [0, 2, 1, 3])
        return output

    # Fallback: backend cannot return LSE (e.g., SDPA on ROCm). Gather KV across
    # the ring group and run local attention to avoid the expensive manual LSE.
    world_size = dist.get_world_size(ring_pg)
    k_bufs = [torch.empty_like(key) for _ in range(world_size)]
    v_bufs = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(k_bufs, key, group=ring_pg)
    dist.all_gather(v_bufs, value, group=ring_pg)
    key_full = torch.cat(k_bufs, dim=2)  # concat along sequence dim
    value_full = torch.cat(v_bufs, dim=2)

    q_bshd = torch.permute(query, [0, 2, 1, 3])
    k_bshd = torch.permute(key_full, [0, 2, 1, 3])
    v_bshd = torch.permute(value_full, [0, 2, 1, 3])

    out = attn_impl.forward(q_bshd, k_bshd, v_bshd, attn_metadata=None)
    return out
