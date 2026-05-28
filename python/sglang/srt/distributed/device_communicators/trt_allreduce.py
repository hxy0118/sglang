"""TRT-LLM style IPC AllReduce for SGLang on AMD ROCm.

Provides high-performance cross-GPU allreduce using IPC shared memory
(bypassing RCCL). Supports CUDAGraph capture/replay and fused
allreduce + residual + RMSNorm.

Usage:
    Loaded automatically by parallel_state.py when running on ROCm with TP > 1.
    Requires trt_allreduce_sglang.so to be importable (either on sys.path or
    in a known location specified by SGLANG_TRT_AR_SO_PATH env var).
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_trt_ar_module = None

ALLREDUCE_SUPPORTED_HIDDEN_SIZES = frozenset({1024, 2048, 2560, 4096, 5120})
ALLREDUCE_FUSION_SUPPORTED_HIDDEN_SIZES = frozenset({1024, 2048, 4096})

FP8_DTYPE = torch.float8_e4m3fnuz
FP8_QUANT_TYPE_ID = 2


def _load_trt_ar_module():
    global _trt_ar_module
    if _trt_ar_module is not None:
        return _trt_ar_module

    so_path = os.environ.get("SGLANG_TRT_AR_SO_PATH", None)
    if so_path:
        so_dir = os.path.dirname(so_path)
        if so_dir not in sys.path:
            sys.path.insert(0, so_dir)

    try:
        import trt_allreduce_sglang
        _trt_ar_module = trt_allreduce_sglang
        return _trt_ar_module
    except ImportError:
        search_paths = [
            "/opt/sglang/python/sglang/srt/distributed/device_communicators",
            "/root/trt_allreduce_sglang",
            os.path.dirname(__file__),
        ]
        for p in search_paths:
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                import trt_allreduce_sglang
                _trt_ar_module = trt_allreduce_sglang
                return _trt_ar_module
            except ImportError:
                continue

    logger.warning(
        "trt_allreduce_sglang.so not found. Set SGLANG_TRT_AR_SO_PATH or "
        "place it in a directory on sys.path."
    )
    return None


class TrtAllReduceComm:
    """IPC shared memory allreduce communicator for AMD ROCm GPUs.

    Manages the lifecycle of CommWorkspace (IPC buffers, barrier flags,
    data workspace) and supports CUDAGraph capture.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    def __init__(
        self,
        group: ProcessGroup,
        device: int,
        max_size_in_bytes: int = 16384 * 16384,
        comm_ptrs_buf_len: int = 1024 * 256,
    ):
        self.group = group
        self.device_id = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_size_in_bytes = max_size_in_bytes
        self.disabled = True
        self._is_capturing = False
        self._capture_pending = False
        self._handle = None

        if self.world_size == 1:
            return
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "TRT allreduce: unsupported world_size=%d", self.world_size
            )
            return

        module = _load_trt_ar_module()
        if module is None:
            return

        torch.cuda.set_device(self.device_id)
        try:
            self._handle = module.TrtArHandle(
                self.device_id, self.rank, self.world_size,
                max_size_in_bytes, comm_ptrs_buf_len,
            )
        except Exception as e:
            logger.warning("TRT allreduce init failed: %s", e)
            return

        barrier_h = self._handle.get_barrier_handle()
        data_h = self._handle.get_data_handle()

        self._barrier()

        barrier_list = [None] * self.world_size
        data_list = [None] * self.world_size
        dist.all_gather_object(barrier_list, barrier_h, group=self.group)
        dist.all_gather_object(data_list, data_h, group=self.group)

        self._handle.open_barrier_handles(barrier_list)
        self._handle.open_data_handles(data_list)

        self._barrier()
        self.disabled = False
        logger.info(
            "TRT allreduce initialized (rank=%d, world_size=%d, device=%d)",
            self.rank, self.world_size, self.device_id,
        )

    def _barrier(self):
        torch.cuda.set_device(self.device_id)
        torch.cuda.synchronize(self.device_id)
        dist.barrier(group=self.group)

    # TRT 1-stage kernel sweet spot: ≤32 tokens (256KB for H=4096 bf16).
    # Beyond this, QR is equally fast or faster.
    # Override with SGLANG_TRT_AR_MAX_BYTES (in bytes).
    _MAX_BYTES_FOR_TRT = int(os.environ.get("SGLANG_TRT_AR_MAX_BYTES", 256 * 1024))

    def should_use(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self._MAX_BYTES_FOR_TRT:
            return False
        hidden_dim = inp.shape[-1]
        if hidden_dim not in ALLREDUCE_SUPPORTED_HIDDEN_SIZES:
            return False
        return True

    def should_use_fused(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        inp_bytes = inp.numel() * inp.element_size()
        if inp_bytes > self.max_size_in_bytes:
            return False
        hidden_dim = inp.shape[-1]
        if hidden_dim not in ALLREDUCE_FUSION_SUPPORTED_HIDDEN_SIZES:
            return False
        return True

    def allreduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(inp)
        if torch.cuda.is_current_stream_capturing():
            self._capture_pending = True
        self._handle.allreduce(inp, out)
        return out

    def allreduce_residual_rmsnorm(
        self,
        allreduce_in: torch.Tensor,
        residual_in: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
        fp8_out: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.cuda.is_current_stream_capturing():
            self._capture_pending = True

        residual_out = torch.empty_like(allreduce_in)
        if fp8_out:
            norm_out = torch.empty_like(allreduce_in, dtype=FP8_DTYPE)
            scale_out = torch.empty(
                allreduce_in.shape[0], 1,
                dtype=torch.float32, device=allreduce_in.device,
            )
            quant_type = FP8_QUANT_TYPE_ID
        else:
            norm_out = torch.empty_like(allreduce_in)
            scale_out = torch.empty(
                1, dtype=torch.float32, device=allreduce_in.device,
            )
            quant_type = 0

        self._handle.allreduce_rms(
            allreduce_in, residual_in, rms_weight,
            residual_out, norm_out, scale_out,
            eps, quant_type,
        )
        return residual_out, norm_out, scale_out

    @contextmanager
    def capture(self):
        """Context manager for CUDAGraph capture mode."""
        try:
            self._is_capturing = True
            yield
        finally:
            self._is_capturing = False
            self._consume_capture()

    def _consume_capture(self):
        if not self._capture_pending:
            return
        self._barrier()
        handles = self._handle.get_captured_handles()
        offsets = self._handle.get_captured_offsets()
        for idx in range(len(handles)):
            handle_list = [None] * self.world_size
            offset_list = [None] * self.world_size
            dist.all_gather_object(handle_list, handles[idx], group=self.group)
            dist.all_gather_object(
                offset_list, int(offsets[idx].item()), group=self.group
            )
            self._barrier()
            self._handle.open_captured_handles(handle_list, offset_list, idx)
        self._handle.capture_clear()
        self._barrier()
        self._capture_pending = False

    def consume_capture_if_needed(self):
        if self._capture_pending:
            self._consume_capture()

    def __del__(self):
        self._handle = None
