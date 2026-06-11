"""Quarantine grader-rejected ONNX and drop in conforming identity baselines.

Context (2026-06-11): Kaggle submission #3 errored with
  "Error processing onnx networks for tasks: [17, 34, 59, 60]"
These four came from an untracked 22:35 batch that bypassed the deploy API.
Root causes (verified locally against neurogolf_utils grader interface):
  * task059 - input hardcoded 11x11, throws on the 30x30 grid the grader feeds
  * task034 - outputs (1,1,30,30) int32 instead of (*,10,30,30) float; undecodable
  * task017 - declared dynamic batch but an internal Mul bakes batch=1 -> throws at N>1
  * task060 - batch dim hardcoded to 1 -> throws at N>1
A single rejected model errors the WHOLE zip (all-or-nothing), so the 69-solved
submission scored nothing while the prior 52-solved one scored 381.02.

task017 and task060 actually solve their public train/test/arc-gen examples at
batch=1 -- they are genuine solves crippled only by the export. They are NOT
salvaged here (their traced graphs bake the batch deep); instead they are
quarantined for Gemini to re-export robustly from solve_017.py / solve_060.py.

Replacement = identity passthrough with input/output [batch,10,height,width]
float32, opset 10 / IR 10 (mirrors task001, a confirmed grader-accepted model).
Dynamic in batch AND spatial dims so it survives whatever the hidden scorer does.
Identity scores 0 on these non-identity tasks -- same as the prior dummy -- but
makes the zip VALID again.
"""
import json
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

WORKING = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
QUARANTINE = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/rescue/grader_rejected_20260611"
MANIFEST = os.path.join(WORKING, "solution_manifest.json")
BROKEN = ["task017", "task034", "task059", "task060"]


def build_identity() -> bytes:
    """input -> Identity -> output, [batch,10,height,width] float32, opset10/IR10."""
    shape = ["batch", 10, "height", "width"]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity_baseline", [x], [y])
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)]
    )
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def verify(blob: bytes) -> None:
    """Identity must pass batch=1, batch=4, and a non-30x30 grid, output==input."""
    sess = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for n, h, w in [(1, 30, 30), (4, 30, 30), (1, 11, 11), (3, 5, 7)]:
        x = np.random.rand(n, 10, h, w).astype(np.float32)
        out = sess.run(["output"], {name: x})[0]
        assert out.shape == (n, 10, h, w), f"shape {out.shape} != {(n,10,h,w)}"
        assert np.array_equal(out, x), "identity output != input"


def main() -> None:
    os.makedirs(QUARANTINE, exist_ok=True)
    blob = build_identity()
    verify(blob)
    print(f"identity baseline built + verified ({len(blob)} bytes)")

    with open(MANIFEST) as f:
        manifest = json.load(f)

    for tid in BROKEN:
        src = os.path.join(WORKING, f"{tid}.onnx")
        if os.path.exists(src):
            shutil.move(src, os.path.join(QUARANTINE, f"{tid}.onnx"))
        with open(src, "wb") as f:
            f.write(blob)
        manifest.setdefault("tasks", {})[tid] = {
            "best_score": None,
            "deployed_score": None,
            "verified_status": "NEEDS_RESOLVE",
            "source_topic": None,
            "created_by": "Gemini",
            "model_sha256": None,
            "model_path": f"neurogolf/data/working/{tid}.onnx",
            "updated_at": "2026-06-11T23:40:00Z",
            "note": "grader-rejected export quarantined; identity baseline; Gemini re-export needed",
        }
        print(f"  {tid}: quarantined -> identity baseline")

    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, MANIFEST)
    print("manifest updated (4 tasks -> NEEDS_RESOLVE / identity)")


if __name__ == "__main__":
    main()
