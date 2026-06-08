# SGLang Qwen3.5 ROCm 上下文并行（CP）移植与优化报告

> 目标：在 SGLang 上让 **Qwen3.5（GQA）跑在 ROCm 后端**，同时支持 prefill CP 与 decode CP，移植 **sgl-project/sglang PR #25090** 的方案到本地 fork。
> 依据：逐文件核验本地 ref `pr25090`（= 近期上游 main v0.5.11 + #25090 已合并）、本地 vLLM（DCP 对照）、相关论文与 PR。
> 调研时间：2026-06-08。

---

## 1. 背景与目标

**Context Parallelism（CP）**：把一条（长）序列的 token 维切到多张 GPU，每卡只持有部分 token 的 KV，通过卡间通信拼出完整 attention。解决两个瓶颈：
- **Prefill 长序列**：attention O(S²)，单卡算力/显存撑不住超长 context。
- **Decode 大 KV**：单序列 KV cache 放不下单卡，或 `tp_size > KV_head 数` 时 TP 复制 KV 浪费显存。

**使能条件**：三家框架都是**静态配置**（无序列长度自动触发），硬约束普遍是 `tp_size % cp_size == 0`。收益场景是长上下文（≥256K 量级）。

**本报告的目标**：把 #25090 的 CP（Triton 后端、AMD 已验证）移植到本地 SGLang，跑通 Qwen3.5 GQA on ROCm 的 prefill + decode CP，并规划后续优化。

---

## 2. 三框架 CP 现状速览（背景，压缩）

| 框架 | CP 实现 | Qwen(GQA) | 阶段 | ROCm | 备注 |
|---|---|---|---|---|---|
| **rtp-llm** | ✅ AllGather / Ring(AllToAll) / AllGather-Overlap，FlashInfer + zigzag | ✅ `CPFlashInferImpl` + Qwen3-Next | prefill | ❌ 仅 CUDA 分支注册，aiter 无 CP | —— |
| **vLLM** | ✅ PCP（ring）+ **DCP**（ag_rs / a2a） | ✅ GQA `_forward_with_dcp` | prefill + **decode** | ❌ DCP 仅 CUDA backend，aiter 无 | DCP 设计=KV 沿 token 分片 |
| **SGLang 上游 main** | PCP（zigzag）+ DSA CP | ❌ 仅 DeepSeek/MLA | prefill | ❌ CUDA-only | —— |
| **SGLang + #25090** | ✅ **DCP（Triton）** 覆盖 decode+prefill | ✅ **GQA** | **prefill + decode** | ✅ **Triton 可移植，MI35x 已测** | **本报告移植对象** |

**结论**：要在 ROCm 上做 Qwen3.5 CP，**#25090 是唯一现成、且本就为 AMD 写的实现**（作者机器即 MI355X，CI 跑 MI35x）。它的关键设计是**走 Triton attention 后端**——Triton 跨 CUDA/ROCm 可移植，且 Triton decode/extend kernel 天然产出 LSE（CP 合并的必需品），从而绕开 aiter/flashmla 在 ROCm 上不返回 LSE 的难题。

---

## 3. #25090 CP 方案详解（移植核心）

### 3.1 先分清：分支上有两套互不相关的 CP 系统

| | **系统 A：DCP**（#25090，移植目标） | **系统 B：PCP / attn_cp**（上游既有，ROCm 死路） |
|---|---|---|
| 开关 | `--dcp-size` / `decode_context_parallel_size` | `--attn-cp-size` / `--enable-prefill-context-parallel` |
| group | `_DCP` / `get_dcp_group()` | `_ATTN_CP` / `get_attention_cp_group()` |
| 切分 | **round-robin** KV token 分片（`token i → rank i % dcp_size`）+ Q-head all-gather + LSE merge | **zigzag** in-seq-split（query 按 `2·cp` 块切） |
| 覆盖 | **decode + prefill/extend 都覆盖** | 仅 prefill |
| 后端 | **Triton 独占**（`triton_backend.py`） | **CUDA FlashAttention fa3 独占** |
| LSE 合并 | `cp_lse_ag_out_rs`（纯 torch + RCCL 集合通信） | `merge_state_v2`（**CUDA kernel**） |
| ROCm | ✅ 0 个 is_hip 阻断 | ❌ FA3 + merge_state_v2 无 ROCm 实现 |

> ⚠️ 命名陷阱：#25090 叫 "DCP"，但其 `_forward_extend_dcp` **把 prefill/extend 也实现了**。所以**一个 `--dcp-size N` 同时给到 prefill-CP 与 decode-CP**。系统 B 是另一回事，移 ROCm 等于重写 kernel（数周），**不作为移植目标**。

**语义说明**：系统 A 的 prefill 是「KV 分片 + Q all-gather」——并行化**分片 prefix KV 的读取**，当前 chunk 的 query 计算各 rank 不切分。系统 B 的 zigzag 才把**新 prompt 的 query 计算**切多卡（真正降单条长 prompt 的 TTFT）。若诉求是「长上下文正确分片 KV + 省显存 + 并行 prefix 读」→ 系统 A 已满足；若诉求是「用 CP 降超长 prompt 的 prefill TTFT」→ 需系统 B（ROCm 重写，见 §4.4 风险）。

### 3.2 系统 A 数据流

**Decode（`TritonAttnBackend.forward_decode` 的 `dcp_size>1` 分支）：**
```
1. q_for_decode = q.view(-1, tp_q_head_num, qk_head_dim)
2. q_for_decode = get_dcp_group().all_gather(q_for_decode, dim=1)   # 聚合各 rank 的 Q head（decode Q 极小，开销小）
3. decode_attention_fwd(q_for_decode, K_shard, V_shard, o_for_decode,
                        kv_indptr, kv_indices, attn_logits, attn_lse, ...)  # 本 rank KV 分片的 partial + LSE
4. local_lse = logsumexp(attn_lse, dim=-1)
5. o = cp_lse_ag_out_rs(o_for_decode, local_lse, group)             # 跨 rank LSE 加权合并 + 切回本 rank head 分片
```

**Prefill / extend（`forward_extend → _forward_extend_dcp`，`dcp_size>1`）：**
```
1. 当前 chunk K/V 仍本地 → extend_attention_fwd_with_lse(..., skip_prefix=True)  得 current_out, current_lse
2. prefix KV 已按 DCP 分片 → q_all = group.all_gather(q_local, dim=1)
   extend_attention_fwd_with_lse(q_all, prefix_K_shard, ..., skip_extend=True) 得各 rank prefix partial + LSE
3. prefix_out, prefix_lse = cp_lse_ag_out_rs(prefix_out, prefix_lse, group, return_lse=True)
4. final_lse = logaddexp(prefix_lse, current_lse)
   out = prefix_out * exp(prefix_lse-final_lse) + current_out * exp(current_lse-final_lse)   # online-softmax 合并
```
> 限制：`_forward_extend_dcp` 对 **sinks / custom_mask / sliding_window 抛 NotImplementedError**（Qwen3.5 full-attn 默认不用，安全）。

**GQA 正确性**：KV head 不分片（保持 per-attn_tp），只把 Q head 在 DCP 子组内 all-gather 给 kernel、算完再 reduce-scatter 回本 rank 的 head 分片 → GQA 组比例（Qwen3.5 16/2=8）天然保持。

### 3.3 KV cache 分片与内存布局

- **token→rank 映射（round-robin）**：写入时 `dcp_kv_mask = positions % dcp_size == dcp_rank`，只在归属 rank 落盘；槽位 `_kv_cache_loc = out_cache_loc // dcp_size`（`triton_backend.py`）。
- **内存池加宽**：`model_runner_kv_cache_mixin.py` 用 `page_size = page_size * dcp_size`、`max_total_num_tokens * dcp_size` 分配 KV pool，使分片后的 token 槽位对齐。
- **每 rank KV 索引（Triton kernel `create_triton_kv_indices_for_dcp_triton`，utils.py）**：
  ```
  first    = kv_start + ((dcp_rank + dcp_size - kv_start%dcp_size) % dcp_size)  # 本 rank 拥有的首个绝对 token
  abs_pos  = first + offset * dcp_size                                          # 按 dcp_size 跨步
  store kv_indices = req_to_token[abs_pos] // dcp_size                          # 映射到本 rank 局部槽位
  ```
  `get_dcp_lens(lens, dcp_size, dcp_rank)` 算各 rank 可见 KV 长度（用于 num_kv_splits 与 indptr）。

### 3.4 LSE 合并原语 `cp_lse_ag_out_rs`（`layers/attention/utils.py`）

```python
def cp_lse_ag_out_rs(cp_attn_out, cp_attn_lse, cp_group, return_lse=False):
    if cp_group.world_size == 1: return cp_attn_out
    lses = cp_group.all_gather(cp_attn_lse, dim=0).view((world_size,) + lse.shape)
    global_lse = logsumexp(lses, dim=0)
    scale = nan_to_num(exp(cp_attn_lse - global_lse)).unsqueeze(-1)
    out = cp_group.all_reduce(nan_to_num(cp_attn_out) * scale)      # 加权求和
    out = out[:, head_start:head_end, :]                            # 切回本 rank TP head（= reduce-scatter 语义）
    return (out, global_lse[:, head_start:head_end]) if return_lse else out
```
- **纯 torch + GroupCoordinator 集合通信**（all_gather + all_reduce + logsumexp/exp）→ 在 RCCL/HIP 上原样可跑，**无任何 CUDA 专用原语**。这是 ROCm 可移植性的根本。
- 数学：每 rank 只算了部分 KV 的 partial softmax，必须用 LSE 做 `exp(lse_i - global_lse)` 加权才能精确合并（online softmax）。decode 与 prefill 用同一原语。

### 3.5 CUDA/HIP graph 与 metadata

- decode metadata：eager 走 `init_forward_metadata → _create_dcp_kv_indices`；cuda-graph 走 `_update_decode_kv_buffers → _fill_dcp_kv_indices` 写入持久 buffer；`num_head = local_heads * dcp_size` 保证 `cuda_graph_attn_logits` buffer 尺寸对。
- **ROCm 注意**：gfx942（MI300X/MI325X）有 `kv_splits ≤ 256` 的 workaround（避免 ~4GiB fp32 buffer 在 ROCm graph replay 下 fault）；仅 gfx950（MI355X）端到端验证过。
- `use_symmetric_memory(group)` 包裹 decode 的 Q all-gather：默认 `enable_symm_mem` 关时应为 nullcontext，RCCL 上需确认。

### 3.6 移植用：关键文件与符号清单

| 文件 | 关键符号 |
|---|---|
| `layers/attention/triton_backend.py` | `_forward_extend_dcp`、`forward_decode` dcp 分支、`_create_dcp_kv_indices`/`_fill_dcp_kv_indices`/`_update_decode_kv_buffers`、`_kv_cache_loc`、`_set_kv_buffer`(dcp_kv_mask)、`num_head=local_heads*dcp_size`、gfx942 cap |
| `layers/attention/utils.py` | `cp_lse_ag_out_rs`、`get_dcp_lens`、`create_triton_kv_indices_for_dcp_triton`、`masked_set_kv_buffer_kernel` |
| `layers/attention/triton_ops/extend_attention.py` | `extend_attention_fwd_with_lse`（`STORE_LSE/SKIP_PREFIX/SKIP_EXTEND` + `_is_hip` 调优） |
| `distributed/parallel_state.py` | `_DCP`、`get_dcp_group`、`decode_context_parallel_size>1` build 块、`tp%dcp==0` 校验 |
| `model_executor/model_runner.py` | `dcp_size`/`dcp_rank` + `initialize_model_parallel(..., decode_context_parallel_size=)` |
| `model_executor/cuda_graph_runner.py` | DCP 捕获/重放 buffer sizing |
| `mem_cache/memory_pool.py` + `model_runner_kv_cache_mixin.py` | 加宽页 KV 分配 + `dcp_kv_mask` 写入 |
| `server_args.py` | `dcp_size`、`--dcp-size`/`--decode-context-parallel-size`、`args.dcp_size = args.decode_context_parallel_size` |
| `model_executor/forward_batch_info.py`、`managers/.../invariant_checker.py` | DCP 元数据 / 计数 |
| `test/registered/amd/test_triton_attention_dcp.py` | MI35x 8 卡 DCP 单测（对标） |

### 3.7 性能预期（为什么值得做）

- **vLLM DCP 实测（同机制，MLA，issue #34018，H200×8，DeepSeek-V2-Lite）**：decode 时延 256K −40% / 512K −58% / **1M −71%** —— 长上下文下提升随 context 增长（KV 读 ÷dcp，通信开销与序列无关）。
- **容量/并发（vLLM PR #24864，Qwen3-235B GQA，TP8→TP8+DCP2）**：**6.57× 最大并发**（262K tok/req），GSM8K 不降。
- **机制公式**：decode attention 是显存带宽瓶颈，`时延 ≈ KV_len·bytes / 带宽`；DCP 让每 rank 只读 `1/dcp` 的 KV → 时延 ÷dcp；通信 `≈ O(B·H)` 与序列无关。**长上下文、MLA/低 KV-head GQA、大 batch 时收益最大；短上下文/batch=1/通信贵时可能回退**。

---

## 4. 移植到本地 SGLang 的计划

**策略：移植系统 A（Triton DCP），不碰系统 B。** DCP 路径无 aiter/CUDA 依赖，基本是干净 cherry-pick，新写代码只有 config guard + 验证。

### 4.1 现状矩阵（Qwen3.5 GQA 视角）

| 能力 | 上游 main | #25090 | GQA | ROCm |
|---|---|---|---|---|
| DCP（decode CP） | ❌ | ✅ 完整 | ✅ | ✅ Triton |
| prefill CP via 系统A（`_forward_extend_dcp`） | ❌ | ✅ 完整 | ✅ | ✅ Triton |
| PCP via 系统B（zigzag） | ✅ | ✅ | ✅ 但 fa3 | ❌ CUDA only |
| aiter 后端（ROCm 默认）CP | ❌ | ❌ | ❌ | ❌ |
| Qwen3.5 GDN 线性层 CP | ❌ | ❌ | ❌ | ❌ |

> Qwen3.5 是**混合模型**：`(l+1) % full_attention_interval == 0` 的层是 full-attention（GQA 16/2，head_dim 256，带 `attn_output_gate` + per-head q/k_norm），其余是 GDN 线性层。**CP 只作用 full-attn 层 KV**；GDN 递归态各 rank 复制。

### 4.2 缺口清单（按风险）

**DCP 侧（小，#25090 已做）：**
1. **[高] aiter+dcp 静默错误**：ROCm 默认后端 aiter 零 CP，`dcp_size>1` 时 KV 不分片、不做 LSE merge → **静默输出错误（不报错）**。无 guard。
2. **[高] 混合模型收益受限**：CP 只覆盖稀疏 full-attn 层，GDN 态复制 → 需先量化收益。
3. **[中] DCP + 投机解码（MTP/EAGLE）= 错**：`speculative/` 无 DCP，Qwen3.5 有 MTP，无 guard。
4. **[中] `_forward_extend_dcp` 不支持 sinks/custom_mask/sliding_window**（Qwen3.5 默认安全）。
5. **[中] ROCm arch**：仅 gfx950 验证；gfx942 有 kv_splits≤256 workaround；其余未验证。
6. **[低]** 每层 all_gather+all_reduce 的 HIP-graph 稳定性 + 性能；`use_symmetric_memory` 在 RCCL 上须确认 no-op；`attn_output_gate`/q/k_norm 在 CP 下组合未验证。

**PCP 侧（大，仅当坚持系统 B 才出现，建议放弃）：** 系统 B GQA PCP 是 FA3-CUDA-only，移 ROCm 要重写 `_fa_cp_attn` + 替换 `merge_state_v2`（HIP/Triton），数周级。**走系统 A 则此缺口消失。**

### 4.3 移植步骤

**Phase 0 — 建基线**：本地已有 `pr25090`。`git diff --stat <fork-base> pr25090 -- <§3.6 文件集>` 看差异；fork 紧跟 main 则优先 `git cherry-pick` #25090 的提交（`dcp init` 等），在 §3.6 文件集解冲突。

**Phase 1 — cherry-pick DCP 核心**：按 §3.6 文件清单逐一搬入并解冲突。

**Phase 2 — 新写 config guard（小，非 kernel）**：
- **G1 [高]**：`server_args._handle_context_parallelism`：`dcp_size>1` 且 `attention_backend != "triton"` → 强制 triton 或报错（杜绝 aiter+dcp 静默错误）。
- **G3 [中]**：同处断言 `speculative_algorithm is None when dcp_size>1`。
- ROCm 默认后端 nudge：Qwen3.5 + `--dcp-size>1` 时选/强制 `--attention-backend triton`。
- （可选）`dcp_size>1` 时断言模型无 sliding_window/sink（干净报错）。

**Phase 3 — ROCm 专项**：确认 `is_gfx942_supported()` 路径并保留 kv_splits≤256 cap；确认 `use_symmetric_memory` 在 RCCL 上是 nullcontext；强制 triton 即绕开所有 aiter/gluon 版本约束。

**Phase 4 — 验证（对标 AMD CI）**：现有 `test/registered/amd/accuracy/mi30x/test_qwen35_eval_amd.py`（Qwen3.5-397B-A17B，TP8，GSM8K，但用 aiter 无 CP）。
1. 复制为 `test_qwen35_dcp_eval_amd.py`，改 `--attention-backend triton --decode-context-parallel-size N`（`tp%dcp==0`），去 `SGLANG_USE_AITER=1`，禁投机。
2. GSM8K 精度对齐 aiter 非-CP 基线（一条 GSM8K 跑 extend→decode，同时验 prefill+decode CP）。
3. **验证顺序**：`cp_lse_ag_out_rs` 单测（dcp=1 必须 == 非 CP）→ 2 卡 smoke（triton dcp=2 vs dcp=1 parity）→ 8 卡 397B（vs aiter baseline parity）。

### 4.4 风险 / 开工前必答

1. **混合模型 CP 收益**：full-attn 稀疏，先量化显存/吞吐再投入。
2. **GDN 线性层不被 seq 分片** → 超长上下文收益有上限；是否要给 GDN 做 sequence-CP（见 §5.3，#25090 之外的大新活）？
3. **gfx 目标**：gfx950（已验证）/ gfx942（仅 workaround）/ 更老（未验证）？
4. **HIP-graph 稳定性**：`cp_lse_ag_out_rs` 每层集合通信在捕获区内，须实测 RCCL。
5. **`attn_output_gate` + q/k_norm 在 CP 下数值正确性**（无测试，须验）。
6. **CP 与投机解码是否需并存**：当前不安全 → 多半互斥。
7. **是否硬性需要系统 B zigzag 语义**（降长 prompt TTFT）→ 若是，ROCm kernel 移植回到桌面。

---

## 5. 后续优化方向

> 移植跑通（正确性）后，按收益/成本排序的优化路线。

### 5.1 通信优化（最高收益）
- **打包 all-to-all 替代 ag+ar**：当前 `cp_lse_ag_out_rs` 每层做 `all_gather(LSE) + all_reduce(out)`（2~3 次集合通信/层）。可借鉴 **vLLM `dcp_alltoall.py` / Helix HOP-B**：把 partial out + fp32 LSE（bitcast 进 2×fp16 槽）打包成单次 `all_to_all_single` + Triton 解包合并，NCCL 3→2，微基准 **1.33×**（vLLM PR #41160）。
- **通信-计算 overlap（HOP-B batchwise）**：把 all-to-all 与 batch 内其余计算 overlap 隐藏。论文消融：MLA(H=1) 通信占比 ~1%、GQA(H=8) ~12% —— **Qwen3.5 GQA 的 overlap 收益更显著**，优先级高。
- **复用你已有的 trtllm/RCCL allreduce 工作**：`cp_lse_ag_out_rs` 的 `all_reduce(out)` 正好可换成你优化过的 trtllm allreduce（前序 P0 工作），small-message 路径直接受益。

### 5.2 ROCm kernel 调优
- **decode/extend Triton kernel 的 gfx 调参**：`waves_per_eu`、`matrix_instr_nonkdim`、`kpack`、`num_kv_splits`、`STORE_TRANSPOSE`；尤其去掉/放宽 gfx942 的 `kv_splits≤256` workaround（定位 ROCm graph replay fault 根因）。
- **aiter 原生 DCP（中长期）**：ROCm 上 aiter PA 通常快于 Triton。若给 aiter decode/extend kernel 加 **LSE 输出**，即可在 aiter_backend 里实现 DCP，拿到比「强制 triton」更好的绝对性能。代价是改 aiter kernel（CK/asm）。
- **fuse LSE rescale 进 attention epilogue**：把 `exp(lse-global_lse)` 缩放融进 decode kernel 尾部，省一趟读写。

### 5.3 覆盖面扩展
- **GDN 线性层 sequence-CP（最大功能缺口，详见 §6）**：Qwen3.5 大部分层是 GDN，#25090 下 GDN 递归态各 rank 复制 → 超长上下文收益受限。这是混合模型 CP 的上限所在，业界方案与现状见 §6。
- **block 级 interleave（`cp_kv_cache_interleave_size>1`）**：当前 round-robin 是 token 级（interleave=1）。引入 vLLM 式 block 级 interleave 可提升页局部性、减少索引开销，并更好地与 prefix-caching/MTP 组合。
- **prefill 真并行（系统 B 语义 on ROCm）**：若确需降超长 prompt 的 TTFT，在 ROCm 上实现 zigzag query-split prefill CP（aiter/Triton 版 `_fa_cp_attn` + HIP `merge_state`）。

### 5.4 鲁棒性与可用性
- **config guard 完善**：dcp×backend、dcp×spec、dcp×sliding_window 的清晰报错（见 §4.2）。
- **prefix caching / chunked prefill 与 DCP 的交互验证**（vLLM 在此处踩过 #26672/#26942 类 bug）。
- **CUDA/HIP graph 下减少每层动态分配**（`q_for_decode.contiguous()`、`o_for_decode=new_empty`、`logsumexp`）—— 预分配持久 buffer，提升 graph 稳定性与吞吐。

---

## 6. GDN / 线性层 sequence-CP：业界方案与现状

> 这是混合模型（Qwen3.5 = GQA full-attn + GDN 线性层）CP 的最大功能缺口。#25090 只 CP 了 full-attn 的 KV，GDN 层在各 rank 复制 → 超长上下文收益受限于此。

### 6.1 核心难点：softmax 的 CP 方案搬不过来

- softmax CP 靠 **LSE-merge**：各 rank 在 KV 分片上独立算 partial softmax，再用 log-sum-exp 精确合并，序列方向**无状态依赖**。
- GDN/线性/Mamba 是**递归**：token *t* 依赖累积状态 `S_t = Σ_{s<t}…`，没有 LSE 式恒等式。按序列切分后，rank *i* 需要 `0..i-1` 产生的**前缀状态** → **要跨 rank 传的是递归状态本身，不是 KV、不是 LSE**。
- 关键性质：递归状态**与序列长度无关**（线性注意力是 `d×d` KV-memory；Mamba/GDN 是 `heads×d_state×d_head`，大小 `O(heads·d_state)`）。这是高效 CP 的可能性来源。

### 6.2 四种业界方案

| 方法 | 传什么 | 通信量 | 模式 | 冗余计算 | 代表 |
|---|---|---|---|---|---|
| **(a) all-gather 激活** | 投影后激活 | `O(seq·hidden)` 随序列 | 1× AllGather | **cp_size×** | **rtp-llm** |
| **(a′) all-to-all 激活重排** | 激活 (z,x,B,C,dt) | `O(seq·hidden/cp)` | 2× All-to-All/方向 | 无 | **Megatron** |
| **(b) ring 状态传递** | 递归状态 S | `(cp-1)·O(heads·d_state)` 序列无关 | 串行 P2P 环 | 无 | **LASP** |
| **(c) 并行 prefix-scan** | chunk 状态 | `cp·O(heads·d_state)` | 1× AllGather / All-Scan | 无 | **LASP-2 / ZeCO** |

- **(a) all-gather**（rtp-llm）：本地投影 → all-gather 拼全序列 → 每 rank 跑整条序列 GDN → 切出本地 token。最简单、数值精确、零风险；代价 cp_size 倍冗余计算 + O(seq) 通信。递归依赖靠"每 rank 都有完整时间轴"本地解决。
- **(a′) Megatron**（`megatron/core/ssm/mamba_context_parallel.py`）：all-to-all 把"序列分片"转"head 分片"，每 rank 拿全序列但只算 1/cp 的 head，**不切递归**。无冗余计算，通信是激活量级。**仅训练，decode 明确不支持**（`assert cp_size==1`）。
- **(b) ring**（LASP, arXiv:2404.02882）：序列切分，`d×d` 状态沿环 P2P 传。通信小且序列无关，但**串行链**，cp 大时延迟瓶颈。
- **(c) prefix-scan**（LASP-2 arXiv:2502.07563 单次 AllGather chunk 状态 + 本地前缀合并；ZeCO arXiv:2507.01004 的 All-Scan）：并行化标准答案，通信既小又无串行依赖。LASP-2 比 LASP 快 15%、比 Ring Attention 快 37%（2048K/64GPU）。**难点：写 Gated DeltaNet 的 chunk 状态结合算子（门控衰减折进前缀合并），易错。**

### 6.3 各框架现状

| 来源 | 线性层 CP | 方法 | 阶段 |
|---|---|---|---|
| **Megatron-Core** | ✅ | (a′) all-to-all | **训练**，decode 不支持 |
| **学术** | ✅ | LASP / LASP-2 / ZeCO | 全是**训练** |
| **vLLM（最新）** | ❌ | CP 仅 softmax(GQA+MLA)；RFC #37995 对线性层用 **batch-split 而非 seq-CP**，明确拒绝 naive token 切分 | —— |
| **SGLang（main+#25090）** | ❌ | CP 仅 softmax；#25090 仅让 GDN "兼容 DCP"(复制+TP)；roadmap #27252 未列线性层 | —— |
| **rtp-llm** | ✅ | **(a) all-gather**，`qwen3_next.py::_forward_cp_prefill` | **prefill-only，CUDA-only**（ROCm 在 `ZigzagProcessor` 硬 fail）|

**结论**：做**推理**的 GDN 层 CP，业界唯一现成实现是 **rtp-llm 的 all-gather（方法 a）**，且 prefill-only + CUDA-only；vLLM/SGLang 上游对线性层一无所有。

### 6.4 对本移植的建议

**第一版采用 rtp-llm 的 all-gather（方法 a）：**
1. **数值精确、零算法风险**：各 rank 跑未改动的 `chunk_gated_delta_rule`，bit-comparable，无需结合算子/前缀合并正确性证明 —— 合"ROCm 先跑通正确性"的目标。
2. **与 #25090 互补**：#25090 是 full-attn 的 **decode** CP，rtp-llm 是 GDN 的 **prefill** all-gather，阶段互补；#25090 已建好 zigzag 布局与 DCP plumbing 可复用。
3. **rtp-llm 的 ROCm 障碍是表面的**：`RTP_LLM_FAIL("not supported on ROCm")` 在 C++ `ZigzagProcessor` 的索引生成里，不在数学里；all-gather 本身是 RCCL 集合通信，ROCm 可跑。要做的是把 zigzag 索引/restore/padding 生成从 C++ 重写到 SGLang Python metadata builder，GDN kernel（`causal_conv1d`/`fused_gdn_gating`/`chunk_gated_delta_rule`）ROCm 上已有，原样复用。

**传什么**：每 GDN 层每次 forward，一次 all-gather **本地投影后的 packed states** `cat([mixed_qkv,b,a])`（在 in_proj 之后、比 hidden_states 小），复用 TP group 作 CP group（`cp_size==tp_size`）；不 gather conv/ssm state（prefill 从零状态起）。**decode 保持复制+TP**（rtp-llm 与 Megatron 都这么 punt）。

**何时升级到 (c) LASP-2**：仅当长上下文/高 CP 度下 cp_size 倍冗余 prefill 计算成瓶颈时。升级路径是把"all-gather 全激活"换成"all-gather 小的 chunk 状态 + 本地前缀合并"，collective 调用点不变，但要写 **Gated DeltaNet 的结合算子**（最难部分）—— profile 驱动，非首版。

**参考实现文件**：`rtp-llm/rtp_llm/models_py/model_desc/qwen3_next.py:699-884`（`_forward_cp_prefill`，移植蓝本）、`rtp_llm/cpp/models/context_parallel/ZigzagProcessor.cc`（zigzag 逻辑，需重写到 Python、去掉 ROCm fail）、`sglang/python/sglang/srt/models/qwen3_5.py`（`Qwen3_5GatedDeltaNet`，目标）、`sglang/.../cp_utils.py` + `triton_backend.py`（#25090 已有 zigzag/DCP 脚手架）。

---

## 7. 参考

**移植对象（pr25090 = main + #25090）：**
- 系统 A DCP：`triton_backend.py`（`_forward_extend_dcp`/`forward_decode` dcp 分支）、`layers/attention/utils.py`（`cp_lse_ag_out_rs`/`get_dcp_lens`/`create_triton_kv_indices_for_dcp_triton`）、`triton_ops/extend_attention.py`（`extend_attention_fwd_with_lse`）、`distributed/parallel_state.py`（`_DCP`/`get_dcp_group`）、`mem_cache/memory_pool.py`、`server_args.py`
- 关键提交：`dcp init`（2026-05-07，作者机器 `smci355…dcgpu` = AMD MI355X）→ `compress dcp path` → `rm reduntant dcp num head param`
- AMD CI：`test/registered/amd/test_triton_attention_dcp.py`（MI35x 8 卡）、`accuracy/mi30x/test_qwen35_eval_amd.py`（对标基线）

**性能与机制（vLLM 对照）：**
- 实测时延：[vLLM issue #34018](https://github.com/vllm-project/vllm/issues/34018)（Helix RFC）；容量/并发：vLLM PR #24864（Qwen3-235B 6.57×）
- 打包 a2a：vLLM PR #41160（1.33×）、`v1/attention/ops/dcp_alltoall.py`；合并原语：`v1/attention/ops/common.py::cp_lse_ag_out_rs`
- 论文：Helix Parallelism [arXiv:2507.07120](https://arxiv.org/abs/2507.07120)（NVIDIA，#25090/vLLM a2a 的理论来源）；ring-CP decode 回退反例 [arXiv:2411.01783](https://arxiv.org/abs/2411.01783)（Meta）

**GDN/线性层 CP（§6）：** Megatron `megatron/core/ssm/mamba_context_parallel.py`（all-to-all，训练）；LASP [arXiv:2404.02882](https://arxiv.org/abs/2404.02882)、LASP-2 [arXiv:2502.07563](https://arxiv.org/abs/2502.07563)、ZeCO [arXiv:2507.01004](https://arxiv.org/abs/2507.01004)；rtp-llm `qwen3_next.py::_forward_cp_prefill`（all-gather，移植蓝本）；vLLM RFC #37995（线性层用 batch-split 非 seq-CP）

**Qwen3.5 模型：** `models/qwen3_5.py`（full-attn `RadixAttention` + GDN 线性层）、`models/qwen3_5_mtp.py`、`configs/qwen3_next.py`（`full_attention_interval`）

**版本快照：** 本地 SGLang fork `chengshu-lcc/sglang@qwen35-0416`（v0.5.9-42）；`pr25090` 基于 v0.5.11；vLLM 本地 2026-03。
