"""Deploy merge-plan winners through the Hub deploy gate, with provenance.

Usage: python3 batch_graft.py [--agent Claude] [--limit N] [--dry-run]
Consumes audits/merge_plan.json (from merge_plan.py). Each entry is POSTed to
the Hub /deploy API, which re-verifies officially, enforces the score gate,
archives the replaced model and rebuilds submission.zip. Successes are logged
to neurogolf/data/working/provenance.json (task -> source bundle/sha/points).
Idempotent: a 409 (incumbent better / already deployed) is logged and skipped.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import requests

AUDITS = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/audits")
PROV = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/provenance.json")
HUB = "http://127.0.0.1:8000/api/project_plugin/neurogolf"


def load_prov():
    return json.loads(PROV.read_text()) if PROV.exists() else {}


def save_prov(p):
    PROV.write_text(json.dumps(p, indent=1, sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="Claude")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = json.loads((AUDITS / "merge_plan.json").read_text())
    if args.limit:
        plan = plan[: args.limit]
    prov = load_prov()

    ok = skip = fail = 0
    gained = 0.0
    t0 = time.time()
    for i, e in enumerate(plan, 1):
        if prov.get(e["task"], {}).get("sha256") == e["sha256"]:
            skip += 1
            continue
        if args.dry_run:
            print(f"DRY {e['task']}: {e['deployed']} -> {e['points']} (+{e['margin']}) from {e['slug']}")
            continue
        with open(e["file"], "rb") as fh:
            r = requests.post(f"{HUB}/deploy",
                              files={"file": (f"{e['task']}.onnx", fh, "application/octet-stream")},
                              data={"task_id": e["task"], "score": f"{e['points']:.3f}",
                                    "agent_name": args.agent},
                              timeout=300)
        if r.status_code == 200:
            ok += 1
            gained += e["margin"]
            prov[e["task"]] = {"source": e["slug"], "sha256": e["sha256"],
                               "points": e["points"], "replaced_points": e["deployed"],
                               "deployed_by": args.agent,
                               "at": datetime.datetime.utcnow().isoformat() + "Z"}
            save_prov(prov)
        elif r.status_code == 409:
            skip += 1
        else:
            fail += 1
            print(f"  FAIL {e['task']} http {r.status_code}: {r.text[:160]}")
        if i % 10 == 0 or i == len(plan):
            rate = (time.time() - t0) / max(1, i)
            print(f"  {i}/{len(plan)}  ok={ok} skip={skip} fail={fail}  +{gained:.1f} pts  "
                  f"({rate:.1f}s/deploy, ~{rate * (len(plan) - i) / 60:.0f} min left)")
    print(f"GRAFT DONE: deployed {ok}, skipped {skip}, failed {fail}, net gain +{gained:.1f} pts")


if __name__ == "__main__":
    main()
