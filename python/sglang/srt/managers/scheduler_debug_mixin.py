import logging

import torch
import triton
import triton.language as tl

from sglang.srt.managers.io_struct import InjectFailureReqInput

logger = logging.getLogger(__name__)


class SchedulerDebugMixin:
    def inject_failure(self, recv_req: InjectFailureReqInput):
        logger.warning(f"Deliberately inject failure (req: {recv_req})")
        # currently only have one failure, but may add more later
        _trigger_oob()


def _trigger_oob():
    x = torch.zeros(1, device="cuda")
    _trigger_oob_kernel[(1,)](x)
    torch.cuda.synchronize()


@triton.jit
def _trigger_oob_kernel(p):
    tl.store(p - 1024, 1.0)
    tl.store(p - 1024 ** 2, 1.0)
