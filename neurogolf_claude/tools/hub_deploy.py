"""Hub deploy helper for NeuroGolf v2 workflow (Claude).

Usage:
    python hub_deploy.py task014 [task021 ...]
    python hub_deploy.py --onnx path/to/model.onnx task014

For each task: officially verify -> extract score -> POST to the Hub /deploy API
(which re-verifies server-side, gates on score, archives, rebuilds submission.zip
and auto-releases the claim). Defaults to solutions/<task>.onnx.
"""
import argparse
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng  # noqa: E402
import onnx  # noqa: E402
import requests  # noqa: E402

ng._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
HUB = "http://192.168.137.215:8000/api/project_plugin/neurogolf"
SOLUTIONS = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"
AGENT = "Claude"


def official(model_path: str, task_num: int):
    """Return (ready, points, summary) by running the official verifier in a scratch dir."""
    scratch = f"/tmp/ng_deploy_{task_num:03d}"
    os.makedirs(scratch, exist_ok=True)
    model = onnx.load(model_path)
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ng.verify_network(model, task_num, ng.load_examples(task_num))
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    ready = "IS READY" in out
    m = re.search(r"yielding ([\d.]+) points", out)
    sz = re.search(r"requires (\d+) bytes \+ (\d+) params", out)
    summary = sz.group(0) if sz else ""
    return ready, (float(m.group(1)) if m else 0.0), summary


def claim(task_id: str, note: str = "SHRINK/SAME family"):
    r = requests.post(f"{HUB}/claim", data={"task_id": task_id, "agent_name": AGENT, "note": note}, timeout=20)
    return r.status_code, r.text


def deploy(task_id: str, onnx_path: str, score: float):
    with open(onnx_path, "rb") as fh:
        files = {"file": (f"{task_id}.onnx", fh, "application/octet-stream")}
        data = {"task_id": task_id, "score": f"{score:.3f}", "agent_name": AGENT}
        r = requests.post(f"{HUB}/deploy", files=files, data=data, timeout=120)
    return r.status_code, r.text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--onnx", default=None, help="explicit onnx path (single task only)")
    ap.add_argument("--no-claim", action="store_true")
    args = ap.parse_args()

    for tid in args.tasks:
        num = int(tid[4:])
        path = args.onnx or os.path.join(SOLUTIONS, f"{tid}.onnx")
        if not os.path.exists(path):
            print(f"{tid}: MISSING {path}")
            continue
        ready, pts, summary = official(path, num)
        if not ready:
            print(f"{tid}: NOT READY — refuse to deploy")
            continue
        print(f"{tid}: READY {pts} pts  {summary}")
        if not args.no_claim:
            sc, _ = claim(tid)
            print(f"  claim -> {sc}")
        sc, txt = deploy(tid, path, pts)
        print(f"  deploy -> {sc}: {txt[:300]}")


if __name__ == "__main__":
    main()
