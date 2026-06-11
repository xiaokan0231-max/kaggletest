"""Officially verify an onnx for a task. Usage: python verify_local.py task310 [path]"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng  # noqa: E402
import onnx  # noqa: E402

ng._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def verify(tid, path=None):
    num = int(tid[4:])
    path = path or f"{SOL}/{tid}.onnx"
    scratch = f"/tmp/ng_verify_{num:03d}"
    os.makedirs(scratch, exist_ok=True)
    m = onnx.load(path)
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ng.verify_network(m, num, ng.load_examples(num))
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    ready = "IS READY" in out
    pts = re.search(r"yielding ([\d.]+) points", out)
    # find per-split pass lines / failures
    fails = re.findall(r"(\d+)/(\d+).*?(?:fail|FAIL)", out)
    return ready, (float(pts.group(1)) if pts else 0.0), out


if __name__ == "__main__":
    tid = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else None
    ready, pts, out = verify(tid, path)
    print(out[-1500:])
    print(f"\n=== {tid}: READY={ready} points={pts} ===")
