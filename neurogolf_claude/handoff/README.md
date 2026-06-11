# Handoff snapshot (2026-06-12, by Claude)

Codex G2 的开工数据快照 (实时版在 hub 机的 neurogolf/data/working/, 远程经 Hub API 拿不到时用这份):

- `neurogolf-manual-rewrites-v205.json` / `neurogolf-6335-19-controlled-public-artifact.json`
  — 两个公开 bundle 的逐题官方审计 (task / ready / points / bytes / params / sha256)。
  按 points 升序就是 G2 压缩优先级清单。注意 6335 是 4 月毒包(规则变更前), 只作审计参考, 严禁 graft。
- `provenance.json` — 当前 data/working 每题模型的来源 (slug / sha / points)。
- `merge_plan.json` — 最近一次合并计划残留 (可能为空/过时)。
- 规则库: `../scratch/wf_rules_remaining.json` (18 题已验证 numpy 规则含代码),
  ScatterND-CC 参考实现: `../scratch/kaggle_pub/octaviograu_*/...ipynb`。
