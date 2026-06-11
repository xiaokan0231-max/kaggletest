"""Force the live deployed set to the best-of TRUSTED audited sources only.

Usage: python3 rebuild_from_trusted.py [--agent Claude] [--dry-run]

Default-deny: reads audits/_poisoned.json, uses ONLY `trusted_slugs`. For every
task, picks the highest-points READY model across trusted audits and deploys it
with allow_regression (overwriting any poison that scored higher locally). Tasks
with no trusted candidate are left as-is and listed. Use this to recover a known-
good state after a poisoned graft. Rewrites provenance for replaced tasks.
"""
import argparse
import datetime
import json
from pathlib import Path

import requests

AUDITS = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/audits")
TRUST_CFG = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/source_trust.json")
PROV = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/provenance.json")
HUB = "http://127.0.0.1:8000/api/project_plugin/neurogolf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="Claude")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((TRUST_CFG).read_text())
    trusted = set(cfg.get("trusted_slugs", []))
    if not trusted:
        raise SystemExit("no trusted_slugs configured")
    print(f"trusted sources: {sorted(trusted)}")

    best = {}
    for af in sorted(AUDITS.glob("*.json")):
        if af.name in ("merge_plan.json", "_poisoned.json"):
            continue
        data = json.loads(af.read_text())
        if data.get("slug") not in trusted:
            continue
        for r in data.get("rows", []):
            if r.get("ready") and (r["task"] not in best or r["points"] > best[r["task"]]["points"]):
                best[r["task"]] = {**r, "slug": data["slug"]}

    status = json.loads(requests.get(f"{HUB}/status", timeout=30).text)
    cur = {t["id"]: t for t in status["tasks"]}
    prov = json.loads(PROV.read_text()) if PROV.exists() else {}

    ok = same = fail = 0
    for task in sorted(best):
        cand = best[task]
        # skip if the live model already IS this exact candidate
        if cur.get(task, {}).get("sha256") == cand["sha256"]:
            same += 1
            continue
        if args.dry_run:
            print(f"DRY {task}: -> {cand['slug']} {cand['points']}")
            continue
        with open(cand["file"], "rb") as fh:
            r = requests.post(f"{HUB}/deploy",
                              files={"file": (f"{task}.onnx", fh, "application/octet-stream")},
                              data={"task_id": task, "score": f"{cand['points']:.3f}",
                                    "agent_name": args.agent, "allow_regression": "true"},
                              timeout=300)
        if r.status_code == 200:
            ok += 1
            prov[task] = {"source": cand["slug"], "sha256": cand["sha256"], "points": cand["points"],
                          "deployed_by": args.agent, "trusted_rebuild": True,
                          "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            PROV.write_text(json.dumps(prov, indent=1, sort_keys=True))
        else:
            fail += 1
            print(f"  FAIL {task} http {r.status_code}: {r.text[:120]}")
        if (ok + fail) % 25 == 0:
            print(f"  ...{ok} replaced, {fail} failed")
    missing = [f"task{i:03d}" for i in range(1, 401) if f"task{i:03d}" not in best]
    print(f"REBUILD DONE: replaced {ok}, already-clean {same}, failed {fail}")
    print(f"tasks with NO trusted candidate ({len(missing)}): {missing[:20]}")


if __name__ == "__main__":
    main()
