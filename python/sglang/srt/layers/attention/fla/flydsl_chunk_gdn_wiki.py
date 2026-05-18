# -*- coding: utf-8 -*-
# FlyDSL Chunk-GDN operator for SGLang — gpu-wiki API backend.
# Calls the gpu-wiki operator (chunk_gdn_flydsl_operator.py) instead of
# maintaining a local megakernel copy.
#
# Enabled by USE_FLYDSL_WIKI=1 environment variable.

import os
import sys
from typing import Optional

import torch
import triton

_WIKI_DIR = os.path.expanduser(
    "~/gpu-wiki/reference-kernels/amd/cdna3/flydsl/FlyDSL"
)
if _WIKI_DIR not in sys.path:
    sys.path.insert(0, _WIKI_DIR)

from chunk_gdn_flydsl_operator import (
    CHUNK_SIZE,
    SUPPORTED_CHUNK_GDN_SHAPES,
    chunk_gdn_flydsl_fwd,
    is_supported_shape,
    make_chunk_offsets,
)

from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.flydsl_chunk_gdn import _kkt_solve
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

RCP_LN2 = 1.0 / 0.6931471805599453

_MIN_T_THRESHOLD = 2048


def is_flydsl_wiki_supported(q: torch.Tensor, v: torch.Tensor, T: int) -> bool:
    _, _, hg, k = q.shape
    _, _, h, dv = v.shape
    shape = (hg, h, k, dv)
    if shape not in SUPPORTED_CHUNK_GDN_SHAPES or T < _MIN_T_THRESHOLD:
        return False
    return True


@torch.compiler.disable
def flydsl_chunk_gdn_wiki_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> tuple:
    """FlyDSL 3-kernel pipeline calling gpu-wiki operator API.

    Returns (g_cumsum, o, A, w, h, v_new) with w=None, v_new=None.
    """
    B, T = q.shape[0], q.shape[1]
    H = beta.shape[2]
    K = q.shape[-1]
    V = v.shape[-1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.float().contiguous()
    beta_bf16 = beta.to(torch.bfloat16)

    state_buffer = initial_state
    if initial_state is not None:
        if initial_state_indices is not None:
            mega_initial_state = initial_state[initial_state_indices].float().contiguous()
        else:
            mega_initial_state = initial_state.float().contiguous()
    else:
        mega_initial_state = None

    if cu_seqlens is not None and cu_seqlens.dtype != torch.long:
        cu_seqlens = cu_seqlens.to(torch.long)

    chunk_indices = None
    chunk_offsets = None
    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)

    g_cumsum = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, scale=RCP_LN2, cu_seqlens=cu_seqlens,
    )

    A = _kkt_solve(
        k=k, g_cumsum=g_cumsum, beta=beta_bf16,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )

    if cu_seqlens is not None:
        total_chunks = len(chunk_indices)
        h_all = q.new_empty(1, total_chunks, H, K, V, dtype=torch.bfloat16)
    else:
        NT = triton.cdiv(T, CHUNK_SIZE)
        h_all = q.new_empty(B, NT, H, K, V, dtype=torch.bfloat16)

    o, final_state = chunk_gdn_flydsl_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g_cumsum=g_cumsum,
        beta=beta_bf16,
        scale=scale,
        initial_state=mega_initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_all=h_all,
    )

    if final_state is not None and state_buffer is not None and initial_state_indices is not None:
        state_buffer[initial_state_indices] = final_state.to(state_buffer.dtype)

    return g_cumsum, o.to(q.dtype), A, None, h_all, None
