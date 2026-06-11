"""Replace every deployed task sourced from a poisoned bundle with the best
alternative from the remaining audits, deploying with allow_regression.

Usage: python3 regraft_source.py <poisoned-slug> [--agent Claude]
For tasks with no READY alternative the poisoned model is left in place and
listed at the end (handle manually: archive restore or dummy).
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
    ap.add_argument("poisoned")
    ap.add_argument("--agent", default="Claude")
    args = ap.parse_args()

    prov = json.loads(PROV.read_text())
    targets = sorted(t for t, v in prov.items() if v["source"] == args.poisoned)
    print(f"{len(targets)} tasks sourced from {args.poisoned}")

    # alternatives from all other audits, never re-selecting a known-poisoned source
    pf = TRUST_CFG
    banned = set(json.loads(pf.read_text()).get("poisoned_slugs", [])) if pf.exists() else set()
    banned.add(args.poisoned)
    alt = {}
    for af in sorted(AUDITS.glob("*.json")):
        if af.name in ("merge_plan.json", "_poisoned.json"):
            continue
        data = json.loads(af.read_text())
        if data.get("slug") in banned:
            continue
        for r in data.get("rows", []):
            if r.get("ready") and (r["task"] not in alt or r["points"] > alt[r["task"]]["points"]):
                alt[r["task"]] = {**r, "slug": data["slug"]}

    ok = miss = fail = 0
    for t in targets:
        cand = alt.get(t)
        if not cand:
            miss += 1
            print(f"  NO ALTERNATIVE for {t} (left poisoned!)")
            continue
        with open(cand["file"], "rb") as fh:
            r = requests.post(f"{HUB}/deploy",
                              files={"file": (f"{t}.onnx", fh, "application/octet-stream")},
                              data={"task_id": t, "score": f"{cand['points']:.3f}",
                                    "agent_name": args.agent, "allow_regression": "true"},
                              timeout=300)
        if r.status_code == 200:
            ok += 1
            prov[t] = {"source": cand["slug"], "sha256": cand["sha256"],
                       "points": cand["points"], "deployed_by": args.agent,
                       "regraft_from": args.poisoned,
                       "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            PROV.write_text(json.dumps(prov, indent=1, sort_keys=True))
        else:
            fail += 1
            print(f"  FAIL {t} http {r.status_code}: {r.text[:120]}")
    print(f"REGRAFT DONE: replaced {ok}, no-alternative {miss}, failed {fail}")


if __name__ == "__main__":
    main()
