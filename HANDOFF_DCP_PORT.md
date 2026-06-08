# DCP 移植交接文档（sglang PR #25090 → 本 fork）

> 目标：把 **Decode Context Parallel (DCP)** 从 `sgl-project/sglang#25090` 移植到本 fork，让 **Qwen3.5 (GQA) 在 ROCm 上跑 prefill+decode CP**。
> 本文档 = 在服务器上**接着移植 + 运行验证**的完整指南。

---

## 0. 现状速览

- **分支**：`dcp-port-qwen35-rocm`
- **WIP commit**：`135b4ee0b`（11 文件已移植，全部 parse 通过，`dcp_size=1` 时行为与原 fork 完全一致 = 安全中间态）
- **策略**：hand-port（fork 与 #25090 跨版本漂移，无干净 cherry-pick range）。已确认 fork 自带基础 CP 框架（System B / attn_cp），**#25090 DCP 自包含、无需补主线 CP**。
- **关键设计**：走 **Triton attention 后端**（ROCm 可移植，且 Triton decode/extend 核天然产出 LSE，绕开 aiter 不返回 LSE 的问题）。详见 `../CP并行支持调研报告.md`。

---

## 1. 把工作搬到服务器

**方式 A（推荐，git）**：本机已 commit。推到 fork 远端后服务器拉取：
```bash
# 本机（或让我帮你 push）：
git push cs dcp-port-qwen35-rocm
# 服务器：
git fetch cs && git checkout dcp-port-qwen35-rocm
```

**方式 B（patch 备份，离线）**：把这些文件 scp 到服务器（在 `/Users/hxy/Desktop/hxy/`）：
- `dcp_port_wip_commit.patch` — 已完成的 WIP commit（`git am < dcp_port_wip_commit.patch`）
- `dcp_25090_full_feature.patch` — **完整 DCP 特性 diff（17 文件，移植剩余部分的总参考）**
- `dcp_remaining_triton_backend.patch` / `dcp_remaining_memory_pool.patch` / `dcp_remaining_pathmapped.patch` — 剩余各文件的 DCP delta
- `dcp_test_triton_attention_dcp.py` / `dcp_test_triton_attention_dcp_utils.py` — #25090 的 AMD DCP 单测

---

## 2. 在服务器上准备 #25090 参考（移植剩余文件必需）

```bash
cd <fork>
git remote add upstream https://github.com/sgl-project/sglang.git   # 若无
git fetch --no-tags --depth=100 upstream pull/25090/head:pr25090
# DCP 特性基线 = pr25090 tip 合并的 main（parent2）：
BASE=$(git rev-list --parents -n1 pr25090 | awk '{print $3}')   # 本机为 2d1856bf4...
echo "BASE=$BASE"
# 重新生成"纯 DCP 特性 patch"（移植剩余文件的权威参考）：
git diff $BASE pr25090 > /tmp/dcp_full_feature.patch
# 查看某文件的 DCP delta：
git diff $BASE pr25090 -- python/sglang/srt/layers/attention/triton_backend.py
```

> 核心原理：`git diff $BASE pr25090` = #25090 在它合并的 main 之上 **纯加的 DCP 代码**。移植剩余文件时，以此为参考，把 DCP 片段织入 fork 对应文件。

---

## 3. 已完成（commit 135b4ee0b，无需再做）

| 文件 | DCP 改动 |
|---|---|
| `distributed/parallel_state.py` | `_DCP` group、`get_dcp_group()`、`initialize_model_parallel(decode_context_parallel_size=)`、`graph_capture` 捕获 `_DCP`、destroy |
| `server_args.py` | `dcp_size` 字段 + `--decode-context-parallel-size`/`--dcp-size` + `args.dcp_size = args.decode_context_parallel_size` |
| `model_executor/model_runner.py` | `self.dcp_size`/`self.dcp_rank` + 传入 `initialize_model_parallel` |
| `model_executor/forward_batch_info.py` | `dcp_kv_mask` 字段 |
| `model_executor/model_runner_kv_cache_mixin.py` | `page_size==1 and dcp_size==1` 门控 → 加宽页分配 |
| `models/utils.py` | DCP 下禁用 fused `set_kv_buffer` 快路径 |
| `mem_cache/common.py` | 用 `allocator.page_size`（DCP 加宽页） |
| `layers/attention/utils.py` | `cp_lse_ag_out_rs`、`get_dcp_lens`、`create_triton_kv_indices_for_dcp_triton` |
| `layers/attention/triton_ops/extend_attention.py` | `extend_attention_fwd_with_lse` + `_fwd_kernel` 的 `STORE_LSE`/`SKIP_PREFIX`/`SKIP_EXTEND`（**已适配 fork 无-k_scale 签名** + 保留 fork block-size 调优） |
| `layers/attention/triton_backend.py` | imports；`__init__` 的 `dcp_size`/`dcp_rank` + `num_head *= dcp_size`；`_dcp_lens`/`_create_dcp_kv_indices`/`_fill_dcp_kv_indices` 助手 |

---

## 4. 剩余移植步骤（按顺序，每步 `python3 -c "import ast;ast.parse(open(F).read())"` 校验）

### 4.1 `triton_backend.py` 收尾（最核心，参考 `dcp_remaining_triton_backend.patch`）
fork 的 cuda-graph 结构与 #25090 不同（fork 用 `init_forward_metadata_{capture,replay}_cuda_graph`，#25090 用 `_update_decode_kv_buffers` 等）。**不能机械套用，必须把 DCP 分支织入 fork 自己的方法体**：

1. **`forward_decode`** 加 `if self.dcp_size > 1:` 分支（pr25090 行 1610-1646 为蓝本）：
   ```
   group = get_dcp_group()
   with use_symmetric_memory(group): q_for_decode = q.view(-1, tp_q_head_num, qk_head_dim).contiguous()
   q_for_decode = group.all_gather(q_for_decode, dim=1).contiguous()
   o_for_decode = q.new_empty((..., layer.v_head_dim)); attn_lse.fill_(-inf)
   self.decode_attention_fwd(q_for_decode, K_shard, V_shard, o_for_decode, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits, ...)
   local_lse = torch.logsumexp(attn_lse[:N,:H,:], dim=-1)
   o = cp_lse_ag_out_rs(o_for_decode, local_lse, group)
   return o.reshape(-1, tp_q_head_num * v_head_dim).to(q.dtype)
   ```
2. **`forward_extend`** → `if self.dcp_size > 1: return self._forward_extend_dcp(...)`；新增 `_forward_extend_dcp`（当前 chunk 本地 `extend_attention_fwd_with_lse(skip_prefix=True)` + 分片 prefix `all_gather(Q)` + `skip_extend=True` + `cp_lse_ag_out_rs` + `logaddexp` 合并）。注意 fork 的 `extend_attention_fwd_with_lse` **已无 k_scale 参数**。
3. **`init_forward_metadata`**（eager，decode 路径）：`dcp_size>1` 时用 `_create_dcp_kv_indices` 建索引；`num_kv_splits_lens = dcp_seq_lens.clamp_min(1)`。
4. **`init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph`**（fork 的）：织入 DCP 索引（用 `_fill_dcp_kv_indices` 写持久 buffer）。这是 cuda-graph DCP 的核心适配点。
5. **`set_kv_buffer` 调用点**（forward_extend ~816、forward_decode ~1020）：`dcp_size>1` 时传 `dcp_kv_mask`；KV 槽位 `out_cache_loc // dcp_size`（若 fork 无 `_kv_cache_loc` 方法则在调用处直接 `// dcp_size`）。

### 4.2 `mem_cache/memory_pool.py`（参考 `dcp_remaining_memory_pool.patch`）
`set_kv_buffer(..., dcp_kv_mask=None)` 形参 + masked write kernel（token 只在归属 rank 落盘）。

### 4.3 路径重映射文件（fork 路径与 #25090 不同，参考 `dcp_remaining_pathmapped.patch`）
| #25090 路径 | fork 路径 |
|---|---|
| `managers/scheduler_components/invariant_checker.py` | `managers/scheduler_runtime_checker_mixin.py` |
| `mem_cache/allocator/paged.py` | `mem_cache/allocator.py` |
| `mem_cache/triton_ops/allocator.py` | **fork 中不存在** —— 确认 fork 的分配器结构后决定是否需要 |

### 4.4 config guard（新写，`server_args._handle_context_parallelism` 或等价处）
- **[高] `dcp_size>1` 强制 `attention_backend=="triton"`**（否则 ROCm 默认 aiter 无 CP → **静默错误**）。
- **[中] `dcp_size>1` 与投机解码互斥**（Qwen3.5 有 MTP；DCP+spec 未实现）。
- 可选：`dcp_size>1` 时模型若有 sliding_window/sink → 干净报错（`_forward_extend_dcp` 不支持）。

---

## 5. 运行验证（在 GPU 服务器上）

```bash
# 0) 重新 build/install fork（按你平时的方式，例如）：
cd <fork> && pip install -e "python[all]"   # 或你的 ROCm build 流程

# 1) parse 兜底（无 GPU 也能跑）：
python3 -c "import ast,glob; [ast.parse(open(f).read()) for f in [
 'python/sglang/srt/layers/attention/triton_backend.py',
 'python/sglang/srt/layers/attention/triton_ops/extend_attention.py',
 'python/sglang/srt/layers/attention/utils.py']]; print('PARSE OK')"

# 2) 单测（#25090 的 DCP 单测，已在 dcp_test_*.py；放到 test/registered/amd/ 后）：
python3 -m pytest test/registered/amd/test_triton_attention_dcp.py -v
python3 -m pytest test/registered/amd/test_triton_attention_dcp_utils.py -v

# 3) 2 卡 smoke（先验正确性，必须 --attention-backend triton）：
python3 -m sglang.launch_server --model <small-qwen3.5-or-gqa> \
  --tp-size 2 --decode-context-parallel-size 2 \
  --attention-backend triton --disable-cuda-graph \
  --trust-remote-code --port 30000
# 起来后发一条 identical-prompt 长上下文请求，对比 --dcp-size 1 输出一致

# 4) 精度对齐（对标现有 AMD CI test_qwen35_eval_amd.py）：
#   复制为 test_qwen35_dcp_eval_amd.py，改 --attention-backend triton + --decode-context-parallel-size N，
#   GSM8K 精度需对齐 aiter 非-CP 基线（rtol 内）。

# 验证顺序：
#   a) cp_lse_ag_out_rs 单测：dcp_size=1 必须 == 非 CP（数值恒等）
#   b) 2 卡：triton dcp=2 vs dcp=1 parity（先 --disable-cuda-graph，过了再开 cuda-graph）
#   c) 8 卡 397B：vs aiter baseline parity
```

**ROCm 注意**：DCP 仅在 `--attention-backend triton` 下生效（aiter 无 DCP）。先 `--disable-cuda-graph` 验证 eager 正确性，再验 cuda-graph（4.1.4 适配后）；gfx942/MI300X 若遇 graph-replay fault，回退 eager 或限制 `triton_attention_num_kv_splits`。

---

## 6. 验收 checklist
- [ ] 4.1 triton_backend 收尾（forward_decode/extend + cuda-graph + set_kv_buffer）
- [ ] 4.2 memory_pool masked write
- [ ] 4.3 路径重映射文件
- [ ] 4.4 config guards
- [ ] 5(a) cp_lse 数值恒等（dcp=1）
- [ ] 5(b) 2 卡 triton dcp=2 vs dcp=1 parity（eager → cuda-graph）
- [ ] 5(c) 8 卡 397B vs aiter baseline parity
- [ ] GDN 线性层 CP（**本期范围外**，见报告 §6：rtp-llm all-gather 法可作后续；当前 GDN 在各 rank 复制，CP 仅覆盖 full-attn 层）

---

## 7. 参考
- 调研与方案全文：`../CP并行支持调研报告.md`（§3 #25090 方案详解、§4 移植计划、§5 优化方向、§6 GDN 线性层 CP）
- #25090：https://github.com/sgl-project/sglang/pull/25090
- DCP 机制对照（vLLM）：issue #34018、PR #24864/#41160；论文 Helix arXiv:2507.07120
