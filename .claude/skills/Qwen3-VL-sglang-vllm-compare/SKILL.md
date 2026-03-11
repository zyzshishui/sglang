---
name: Qwen3-VL-sglang-vllm-compare
description: Systematic workflow for benchmarking SGLang VLM serving against vLLM, profiling with torch traces, identifying bottlenecks, and applying optimizations. Use when SGLang is slower than vLLM on Qwen3-VL-8B-Instruct and you need to close the gap.
---

# VLM Performance Analysis & Optimization: SGLang vs vLLM

Systematic guide for diagnosing and fixing VLM (Vision Language Model) serving performance regressions in SGLang relative to vLLM. Covers benchmarking, profiling, trace analysis, and iterative optimization.

## When to Use This Skill

- SGLang VLM throughput and latency is significantly worse than vLLM on the same model/hardware
- You need to profile SGLang and vLLM side-by-side to identify bottlenecks
- You want to port or adapt a vLLM optimization to SGLang
- You're investigating req/s, TTFT, TPOT, ITL, or throughput regressions on vision-language models

## Prerequisites

- SGLang repo at `/root/sglang`, editable install
- vLLM repo at `/root/vllm` for reference
- Model weights accessible (e.g. `/workspace/Qwen3-VL-8B-Instruct`)
- Shared storage at `/workspace` accessible from both SGLang and vLLM containers (if using separate containers)
- GPU with sufficient VRAM for the target model

---

## Phase 1: Establish Baseline Benchmarks

### 1.1 Launch SGLang Server

```bash
sglang serve \
    --model-path /workspace/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --attention-backend triton
```

### 1.2 Launch vLLM Server(not in this container)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code
```

### 1.3 Run Benchmark (Same Workload for Both)

Use SGLang's `bench_serving` for consistent measurement across backends:

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --model /workspace/Qwen3-VL-8B-Instruct \
    --dataset-name image \
    --num-prompts 500 \
    --random-input-len 256 \
    --random-output-len 256 \
    --random-range-ratio 1.0 \
    --image-count 1 \
    --image-resolution 720p \
    --image-content random \
    --request-rate inf \
    --warmup-requests 10
```

For vLLM, change `--backend vllm`

### 1.4 Key Metrics to Compare

| Metric | What It Tells You |
|--------|-------------------|
| **Request throughput (req/s)** | Overall system capacity |
| **Input token throughput (tok/s)** | Prefill-phase efficiency |
| **Output token throughput (tok/s)** | Decode-phase efficiency |
| **TTFT (ms)** | Prefill latency (includes vision encoding + first LLM forward) |

---

## Phase 2: Profile Both Systems

### 2.1 Profile SGLang

**Option A: Via bench_serving (recommended)**

```bash
python3 -m sglang.bench_serving \
    --backend sglang \
    --model /workspace/Qwen3-VL-8B-Instruct \
    --dataset-name image \
    --num-prompts 100 \
    --random-input-len 256 \
    --random-output-len 256 \
    --random-range-ratio 1.0 \
    --image-count 1 \
    --image-resolution 720p \
    --image-content random \
    --request-rate inf \
    --warmup-requests 10 \
    --profile \
    --profile-output-dir /root/sglang/trace \
    --profile-by-stage \
    --profile-stages prefill decode \
    --profile-num-steps 30
```

This generates separate EXTEND (prefill) and DECODE traces as `.trace.json.gz`.

**Option B: Via API**

```bash
# Start profiling
curl -X POST http://localhost:30000/start_profile

# ... run workload ...

# Stop profiling
curl -X POST http://localhost:30000/stop_profile
```

### 2.2 Profile vLLM (cannot be executed in this container, so already launched and accessible within this container, never kill vLLM!!!)

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /workspace/Qwen3-VL-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/workspace/g23/vllm/traces \
    --profiler-config.max_iterations=30
```

Then trigger workload via benchmark. Traces go to `--profiler-config.torch_profiler_dir`.

> **Important:** Profiling adds overhead and distorts timing numbers. Do NOT use profiled runs for throughput/latency comparisons. Use profiling only for structural analysis (which kernels run, call stacks, relative proportions).

### 2.3 Analyze Traces

Open traces in [Perfetto UI](https://ui.perfetto.dev):

1. Upload `.trace.json.gz` (supports gzip directly)
2. Compare GPU kernel timelines side-by-side
---

## Phase 3: Identify Bottlenecks

### 3.1 Architecture Comparison

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Vision encoder execution** | Runs inside `model.forward()` during extend/prefill | Runs separately in `_execute_mm_encoder()` before LLM prefill |
| **Encoder caching** | Precomputed embeddings optional; no hash-based cache manager | `EncoderCacheManager` with hash-based dedup, shared across requests |
| **Encoder scheduling** | Part of main scheduler batch | Separate scheduler step: `_determine_encoder_inputs_to_schedule()` |
| **Chunked prefill for MM** | Supported; vision encoder may run per chunk | Encoder inputs scheduled per chunk; embeddings gathered per chunk |
| **Vision encoder DP** | `run_dp_sharded_mrope_vision_model()` in `mm_utils` | `run_dp_sharded_mrope_vision_model()` in `vision.py`, load-balanced by patch count |
| **MRoPE positions** | Computed in `forward_batch_info.py` | Computed in `mrope_utils.py` |
| **Attention backends** | triton, flashinfer, flashattention, aiter, wave, nsa | FlashAttention, FlashInfer, Triton, CuDNN |
| **Video optimization** | — | EVS (efficient video sampling) with cosine similarity pruning |

---

## Phase 4: Iterative Optimization

### 4.1 Optimization Loop

```
┌─────────────────────────────────────────┐
│ 1. Identify bottleneck from trace       │
│    (vision encoder? attention? sched?)  │
├─────────────────────────────────────────┤
│ 2. Study vLLM's approach for that area  │
│    (read code, compare trace timings)   │
├─────────────────────────────────────────┤
│ 3. Implement optimization in SGLang     │
├─────────────────────────────────────────┤
│ 4. Re-profile (with tracing, no perf   │
│    numbers from profiled run)           │
├─────────────────────────────────────────┤
│ 5. Re-benchmark (without tracing, for  │
│    accurate perf numbers)              │
├─────────────────────────────────────────┤
│ 6. Compare metrics → repeat if needed   │
└─────────────────────────────────────────┘
```

### 4.2 Re-Profile After Each Change

```bash
# Profile (for structural analysis only)
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name image \
    --num-prompts 100 \
    --request-rate inf \
    --profile \
    --profile-output-dir /root/sglang/trace/ \
    --profile-by-stage \
    --profile-stages prefill decode \
    --profile-num-steps 30 \
    ... # same workload params

# Benchmark (for accurate numbers, NO profiling)
python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name image \
    --num-prompts 500 \
    --request-rate inf \
    ... # same workload params
```

### 4.4 Triggering vLLM Re-Profile (Remote Container)

If vLLM is running in a separate container with `--profiler-config.profiler=torch`:

```bash
# Send workload to trigger profiling
cd /root/sglang/python && python3 -m sglang.bench_serving \
    --backend vllm \
    --model /workspace/Qwen3-VL-8B-Instruct \
    --dataset-name image \
    --num-prompts 100 \
    --request-rate inf \
    --warmup-requests 10 \
    ...
```

Traces will appear in the configured `--profiler-config.torch_profiler_dir`.

---

## Phase 5: Validate & Report

### 5.1 Before/After Comparison Table

```markdown
| Metric | SGLang (before) | SGLang (after) | vLLM (baseline) |
|--------|-----------------|----------------|-----------------|
| Duration (s) | | | |
| Request throughput (req/s) | | | |
| Output throughput (tok/s) | | | |
| TTFT mean (ms) | | | |
| TPOT mean (ms) | | | |
| ITL P99 (ms) | | | |
```
