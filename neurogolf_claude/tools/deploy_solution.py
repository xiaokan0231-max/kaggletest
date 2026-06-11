"""Score-gated deployment for neurogolf solutions (forum #47 proposal).

Usage: python deploy_solution.py <candidate.onnx> [taskNNN]

- Officially verifies the candidate (must print IS READY).
- If the deployed model in data/working is a real solution (not an 868-byte
  dummy), verifies it too and refuses to deploy a lower-scoring candidate.
- Archives the replaced version to data/working/archive/ named with its score.
- Deploys as a single self-contained file (external data embedded) and removes
  orphaned .onnx.data sidecars.
- Reminds you to rebuild submission.zip.
"""
import contextlib
import io
import os
import re
import shutil
import sys

sys.path.insert(0, "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils")
import neurogolf_utils as ng  # noqa: E402
import onnx  # noqa: E402

ng._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
WORKING = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
DUMMY_SIZE = 868


def official_points(model_path: str, task_num: int):
    """Returns (ready, points). Runs in a scratch dir; verify_network writes files to CWD."""
    scratch = f"/tmp/ng_deploy_{task_num:03d}"
    os.makedirs(scratch, exist_ok=True)
    model = onnx.load(model_path)  # resolves any external data relative to model_path
    cwd = os.getcwd()
    os.chdir(scratch)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ng.verify_network(model, task_num, ng.load_examples(task_num))
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    m = re.search(r"yielding ([\d.]+) points", out)
    return "IS READY" in out, float(m.group(1)) if m else 0.0


def main() -> None:
    candidate = sys.argv[1]
    name = os.path.basename(candidate)
    task_num = int(sys.argv[2][4:]) if len(sys.argv) > 2 else int(re.search(r"task(\d{3})", name).group(1))
    target = os.path.join(WORKING, f"task{task_num:03d}.onnx")

    ready, cand_pts = official_points(candidate, task_num)
    if not ready:
        sys.exit(f"REFUSED: candidate for task{task_num:03d} is not IS READY.")
    print(f"candidate task{task_num:03d}: {cand_pts} points")

    if os.path.exists(target) and os.path.getsize(target) != DUMMY_SIZE:
        inc_ready, inc_pts = official_points(target, task_num)
        print(f"incumbent task{task_num:03d}: ready={inc_ready}, {inc_pts} points")
        if inc_ready and inc_pts >= cand_pts:
            sys.exit(f"REFUSED: incumbent scores {inc_pts} >= candidate {cand_pts}. Keep the better model.")
        archive = os.path.join(WORKING, "archive")
        os.makedirs(archive, exist_ok=True)
        emb = onnx.load(target)
        onnx.save(emb, os.path.join(archive, f"task{task_num:03d}_{inc_pts:.3f}pts.onnx"))
        print(f"archived incumbent -> archive/task{task_num:03d}_{inc_pts:.3f}pts.onnx")

    onnx.save(onnx.load(candidate), target)  # embeds external data if any
    sidecar = target + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
        print(f"removed orphaned sidecar {os.path.basename(sidecar)}")
    print(f"deployed task{task_num:03d} ({cand_pts} points). Now run rebuild_submission.py.")


if __name__ == "__main__":
    main()
