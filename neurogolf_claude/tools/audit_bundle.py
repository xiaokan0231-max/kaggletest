"""Officially audit every taskNNN.onnx in a bundle, in parallel.

Usage: python3 audit_bundle.py <slug> [--workers 12] [--shift auto|0|1|-1]

For each model runs the OFFICIAL verifier (neurogolf_utils.verify_network) and
records {task, ready, points, bytes, params, sha256}. Resumable: existing rows
in the output JSON are kept. Output: neurogolf/data/working/audits/<slug>.json

Index-shift hazard: some bundles are 0-indexed (file task000 == our task001).
With --shift auto (default) the audit samples 12 tasks at shift 0; if fewer
than 15% verify READY it re-samples at +1 and -1 and picks the best shift for
the full run. The chosen shift is recorded in the JSON header.
"""
import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

BUNDLES = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/bundles")
AUDITS = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/audits")
RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"


def _verify_one(job):
    """Worker: officially verify one model file against one task number."""
    path, task_num, scratch_root = job
    sys.path.insert(0, RAW + "neurogolf_utils")
    import neurogolf_utils as ng  # noqa: PLC0415
    import onnx  # noqa: PLC0415
    ng._NEUROGOLF_DIR = RAW
    scratch = Path(scratch_root) / f"t{task_num:03d}_{os.getpid()}"
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
                        "error": str(exc)[:120], "sha256": sha, "file": str(path)}
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    ready = "IS READY" in out
    pts = re.search(r"yielding ([\d.]+) points", out)
    sz = re.search(r"requires (\d+) bytes \+ (\d+) params", out)
    return {"task": f"task{task_num:03d}", "ready": ready,
            "points": float(pts.group(1)) if pts else 0.0,
            "bytes": int(sz.group(1)) if sz else None,
            "params": int(sz.group(2)) if sz else None,
            "sha256": sha, "file": str(path)}


def jobs_for_shift(files, shift, scratch_root):
    jobs = []
    for f in files:
        n = int(re.match(r"task(\d+)\.onnx$", f.name).group(1)) + shift
        if 1 <= n <= 400:
            jobs.append((str(f), n, scratch_root))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--workers", type=int, default=max(2, mp.cpu_count() - 3))
    ap.add_argument("--shift", default="auto")
    args = ap.parse_args()

    bdir = BUNDLES / args.slug
    files = sorted(bdir.glob("task*.onnx"))
    if not files:
        sys.exit(f"no task onnx in {bdir} — run bundle_pull.py first")
    AUDITS.mkdir(parents=True, exist_ok=True)
    out_path = AUDITS / f"{args.slug}.json"
    scratch_root = f"/tmp/ng_audit/{args.slug}"

    # ---- determine shift ----
    if args.shift == "auto":
        shift = 0
        sample = files[:: max(1, len(files) // 12)][:12]
        with mp.Pool(min(args.workers, len(sample))) as pool:
            r0 = pool.map(_verify_one, jobs_for_shift(sample, 0, scratch_root))
        rate0 = sum(r["ready"] for r in r0) / max(1, len(r0))
        print(f"shift 0 sample ready-rate: {rate0:.0%}")
        if rate0 < 0.15:
            rates = {0: rate0}
            for s in (1, -1):
                with mp.Pool(min(args.workers, len(sample))) as pool:
                    rs = pool.map(_verify_one, jobs_for_shift(sample, s, scratch_root))
                rates[s] = sum(r["ready"] for r in rs) / max(1, len(rs))
                print(f"shift {s:+d} sample ready-rate: {rates[s]:.0%}")
            shift = max(rates, key=rates.get)
        print(f"using shift {shift:+d}")
    else:
        shift = int(args.shift)

    # ---- resume ----
    done = {}
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("shift") == shift:
            done = {r["task"]: r for r in prev.get("rows", [])}
            print(f"resuming: {len(done)} rows already audited")

    jobs = [j for j in jobs_for_shift(files, shift, scratch_root)
            if f"task{j[1]:03d}" not in done]
    print(f"auditing {len(jobs)} models with {args.workers} workers ...")
    rows = list(done.values())
    if jobs:
        with mp.Pool(args.workers, maxtasksperchild=20) as pool:
            for i, r in enumerate(pool.imap_unordered(_verify_one, jobs), 1):
                rows.append(r)
                if i % 25 == 0 or i == len(jobs):
                    ready = sum(x["ready"] for x in rows)
                    out_path.write_text(json.dumps(
                        {"slug": args.slug, "shift": shift, "rows": rows}, indent=1))
                    print(f"  {i}/{len(jobs)} done (total READY {ready})")
    out_path.write_text(json.dumps({"slug": args.slug, "shift": shift, "rows": rows}, indent=1))
    ready = [r for r in rows if r["ready"]]
    print(f"DONE {args.slug}: {len(ready)}/{len(rows)} READY, "
          f"sum {sum(r['points'] for r in ready):.1f} pts -> {out_path}")


if __name__ == "__main__":
    main()
