"""Compute the best-of merge plan across all audited bundles vs current Hub state.

Usage: python3 merge_plan.py [--epsilon 0.001]
Reads every neurogolf/data/working/audits/<slug>.json plus the Hub /status,
picks per task the READY candidate with the highest official points, and emits
audits/merge_plan.json sorted by margin (descending) for batch_graft.py.
"""
import argparse
import json
import urllib.request
from pathlib import Path

AUDITS = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/audits")
TRUST_CFG = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/source_trust.json")
HUB = "http://127.0.0.1:8000/api/project_plugin/neurogolf"


def trust_cfg():
    return json.loads(TRUST_CFG.read_text()) if TRUST_CFG.exists() else {}


def source_allowed(slug: str, cfg: dict) -> bool:
    """Default-deny: in allowlist mode only trusted_slugs may graft the live
    submission (candidate/poison sources need separate single-source LB tests)."""
    if cfg.get("mode") == "allowlist":
        return slug in set(cfg.get("trusted_slugs", []))
    return slug not in set(cfg.get("poisoned_slugs", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epsilon", type=float, default=0.001)
    args = ap.parse_args()

    status = json.load(urllib.request.urlopen(f"{HUB}/status", timeout=30))
    deployed = {}
    for t in status["tasks"]:
        deployed[t["id"]] = float(t.get("deployed_score") or 0.0) if t.get("solved") else 0.0

    cfg = trust_cfg()
    best = {}  # task -> candidate row + slug
    for af in sorted(AUDITS.glob("*.json")):
        if af.name in ("merge_plan.json", "_poisoned.json"):
            continue
        data = json.loads(af.read_text())
        if not source_allowed(data.get("slug", ""), cfg):
            continue
        for r in data.get("rows", []):
            if not r.get("ready"):
                continue
            cur = best.get(r["task"])
            if cur is None or r["points"] > cur["points"]:
                best[r["task"]] = {**r, "slug": data["slug"]}

    plan, kept = [], 0
    for task, cand in sorted(best.items()):
        have = deployed.get(task, 0.0)
        margin = cand["points"] - have
        if margin > args.epsilon:
            plan.append({"task": task, "slug": cand["slug"], "file": cand["file"],
                         "points": round(cand["points"], 3), "deployed": round(have, 3),
                         "margin": round(margin, 3), "sha256": cand["sha256"]})
        else:
            kept += 1
    plan.sort(key=lambda x: -x["margin"])

    out = AUDITS / "merge_plan.json"
    out.write_text(json.dumps(plan, indent=1))
    total_margin = sum(p["margin"] for p in plan)
    cur_total = sum(deployed.values())
    mode = cfg.get("mode", "denylist")
    allowed = sorted(cfg.get("trusted_slugs", [])) if mode == "allowlist" else f"all but {sorted(cfg.get('poisoned_slugs', []))}"
    print(f"source policy: {mode} -> {allowed}")
    print(f"current deployed total : {cur_total:.1f} pts ({sum(1 for v in deployed.values() if v > 0)} solved)")
    print(f"merge plan             : {len(plan)} replacements, ours kept where better/equal: {kept}")
    print(f"projected gain         : +{total_margin:.1f}  ->  {cur_total + total_margin:.1f} pts")
    print(f"plan written to {out}")


if __name__ == "__main__":
    main()
