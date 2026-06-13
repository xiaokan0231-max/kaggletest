"""Officially audit the LIVE deployed models in neurogolf/data/working/task*.onnx.

This is the TRUE current submission state — runs the official verifier on every
deployed model and records {task, ready, points, bytes, params, sha256}.
Output: neurogolf/data/working/audits/_WORKING.json
"""
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

WORKING = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working")
AUDITS = WORKING / "audits"
RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"


def _verify_one(path):
    sys.path.insert(0, RAW + "neurogolf_utils")
    import neurogolf_utils as ng  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    ng._NEUROGOLF_DIR = RAW
    task_num = int(re.match(r"task(\d+)\.onnx$", Path(path).name).group(1))
    scratch = Path(f"/tmp/ng_audit_working/t{task_num:03d}_{os.getpid()}")
    scratch.mkdir(parents=True, exist_ok=True)
    blob = Path(path).read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                ng.verify_network(onnx.load(str(path)), task_num, ng.load_examples(task_num))
            except Exception as exc:  # noqa: BLE001
                return {"task": f"task{task_num:03d}", "ready": False, "points": 0.0,
                        "error": str(exc)[:140], "filesize": len(blob), "sha256": sha}
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    ready = "IS READY" in out
    pts = re.search(r"yielding ([\d.]+) points", out)
    sz = re.search(r"requires (\d+) bytes \+ (\d+) params", out)
    err = ""
    if not ready:
        m = re.search(r"Error:[^\n]+", out)
        err = m.group(0)[:140] if m else out.strip()[-140:]
    return {"task": f"task{task_num:03d}", "ready": ready,
            "points": float(pts.group(1)) if pts else 0.0,
            "bytes": int(sz.group(1)) if sz else None,
            "params": int(sz.group(2)) if sz else None,
            "filesize": len(blob), "error": err, "sha256": sha}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(2, mp.cpu_count() - 2)
    files = sorted(str(p) for p in WORKING.glob("task*.onnx"))
    AUDITS.mkdir(parents=True, exist_ok=True)
    out_path = AUDITS / "_WORKING.json"
    print(f"auditing {len(files)} deployed models with {workers} workers ...")
    rows = []
    with mp.Pool(workers, maxtasksperchild=15) as pool:
        for i, r in enumerate(pool.imap_unordered(_verify_one, files), 1):
            rows.append(r)
            if i % 40 == 0 or i == len(files):
                ready = sum(x["ready"] for x in rows)
                pts = sum(x["points"] for x in rows)
                out_path.write_text(json.dumps({"rows": rows}, indent=1))
                print(f"  {i}/{len(files)} | READY {ready} | sum {pts:.1f}")
    out_path.write_text(json.dumps({"rows": rows}, indent=1))
    ready = [r for r in rows if r["ready"]]
    failed = [r for r in rows if not r["ready"]]
    print(f"\nDONE: {len(ready)}/{len(rows)} READY, true sum {sum(r['points'] for r in ready):.1f} pts")
    print(f"FAILING ({len(failed)}): scoring 0 on real grader")
    for r in sorted(failed, key=lambda x: x["task"]):
        print(f"  {r['task']}: filesize={r.get('filesize')} {r.get('error','')[:90]}")


if __name__ == "__main__":
    main()
