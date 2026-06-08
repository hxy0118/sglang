# 服务器开工 Prompt（直接贴给服务器上的 Claude Code）

> 用法：在服务器上 `cd` 到 fork 仓库根目录，启动 Claude Code，把下面 `===` 之间的整段贴进去。

```
===
我在把 sglang PR #25090 的 Decode Context Parallel (DCP) 移植到本 fork，目标是让 Qwen3.5 (GQA)
在 ROCm 后端跑 prefill+decode CP。已有一个进行到一半的分支，现在要在这台 GPU 服务器上接着移植 + 验证。

第一步，请先读这两个文件，它们是完整的交接与方案文档：
- HANDOFF_DCP_PORT.md     ← 交接：现状、剩余步骤(精确到方法+pr25090 行号)、验证命令、验收 checklist
- CP并行支持调研报告.md    ← 方案：§3 #25090 详解 / §4 移植计划 / §5 优化 / §6 GDN 线性层 CP

背景事实（已验证，照做即可）：
- 当前分支 dcp-port-qwen35-rocm 已 commit 了基础层 9 文件 + extend_attention.py 核函数 + triton_backend.py
  的 imports/__init__/3 个 DCP 索引助手；全部 parse 通过，dcp_size=1 时行为与原 fork 完全一致。
- fork 已自带基础 CP 框架(System B/attn_cp)，#25090 的 DCP 自包含、无需补主线 CP。
- 关键设计：走 Triton attention 后端(ROCm 可移植、decode/extend 核天然产出 LSE)。
- extend_attention.py 已适配 fork 的"无 k_scale/v_scale"签名，别再引入 k_scale。

请先执行这些命令准备 #25090 参考(移植剩余文件的权威来源)：
  git fetch --no-tags --depth=100 upstream pull/25090/head:pr25090
  BASE=$(git rev-list --parents -n1 pr25090 | awk '{print $3}')
  echo "BASE=$BASE"   # 纯 DCP 特性 = git diff $BASE pr25090
  # 看某文件 DCP delta：git diff $BASE pr25090 -- <path>

然后按 HANDOFF_DCP_PORT.md §4 顺序移植剩余部分（每步 ast.parse 校验）：
  4.1 triton_backend.py 收尾（最核心，必须织入 fork 自己的方法体，不能机械套用 pr25090）：
      forward_decode 的 dcp_size>1 分支(all_gather Q + decode + cp_lse_ag_out_rs；pr25090 L1610-1646)、
      forward_extend→_forward_extend_dcp、init_forward_metadata decode 索引钩子、
      cuda-graph capture/replay 适配(fork 用 init_forward_metadata_{capture,replay}_cuda_graph)、
      set_kv_buffer(dcp_kv_mask) + out_cache_loc//dcp_size。
  4.2 mem_cache/memory_pool.py：set_kv_buffer(dcp_kv_mask=None) + masked write kernel。
  4.3 路径重映射文件：invariant_checker→managers/scheduler_runtime_checker_mixin.py；
      allocator/paged.py→mem_cache/allocator.py；triton_ops/allocator.py 在 fork 不存在(确认后再定)。
  4.4 config guard：dcp_size>1 强制 --attention-backend triton(否则 ROCm 默认 aiter 会静默出错)；
      dcp_size>1 与投机解码互斥。

验证(GPU 上，HANDOFF §5)：
  a) python3 -m pytest test/registered/amd/test_triton_attention_dcp_utils.py -v   # cp_lse 数值恒等，现在就能跑
  b) 2 卡 smoke：--tp-size 2 --decode-context-parallel-size 2 --attention-backend triton --disable-cuda-graph
     对比 --dcp-size 1 输出一致(先 eager，过了再开 cuda-graph)
  c) 8 卡 397B：对标 test_qwen35_eval_amd.py，GSM8K 精度对齐 aiter 非-CP 基线

请用 TodoWrite 维护 HANDOFF §6 的验收 checklist，逐项推进；每改一个文件先 ast.parse，
改完一组就让我在 GPU 上验证后再继续。先从读 HANDOFF + 准备 pr25090 参考开始。
===
```

## 备注
- 若 `upstream` 远端不存在：`git remote add upstream https://github.com/sgl-project/sglang.git`
- 本机 `/Users/hxy/Desktop/hxy/` 下还有离线参考 patch（`dcp_25090_full_feature.patch` 等），可 scp 过来；但更推荐用上面的 `git fetch pr25090` 在服务器现生成。
- 分支远端：`hxy` = github.com/hxy0118/sglang（已 push）。
