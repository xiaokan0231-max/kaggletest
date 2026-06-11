"""Rebuild neurogolf submission.zip from data/working/task*.onnx.

Two hazards this guards against (found 2026-06-11):
1. Stale zip: solutions exported after the last zip build stay dummies inside.
2. External data: models saved with `task*.onnx.data` sidecars break when the
   zip ships only the .onnx (the scorer cannot resolve the external tensors).

Every model is re-saved with all tensors embedded, then zipped. Run after any
harvest that touches data/working.
"""
import glob
import io
import os
import sys
import zipfile

import numpy as np
import onnx
import onnxruntime as ort

WORKING = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
ZIP_PATH = os.path.join(WORKING, "submission.zip")


def inference_check(blob: bytes) -> str:
    """Run the model the way the Kaggle grader does and return '' if healthy.

    The grader (see neurogolf_utils.convert_to_numpy/run_network) feeds a
    batch=1 one-hot float32 tensor [1,10,30,30] and reads a float output it
    thresholds at 0 then decodes channel-wise. onnx.checker passing is NOT
    enough -- task034/059 passed the checker yet errored the whole 2026-06-11
    submission (1-channel int32 output, hardcoded 11x11 input).

    The output grid may be ANY H x W (that is how SHRINK/EXPAND tasks resize),
    so we only require: runs at batch=1, rank-4 output, channel dim == 10, float
    dtype. A model that throws or violates that contract poisons the entire zip
    (scoring is all-or-nothing), so this is a hard gate.

    Caveat: this cannot catch models that work at batch=1 but throw at batch>1
    (task017/060). They are indistinguishable here from the ~380 batch=1 dummies
    the grader accepts, so that class is caught only by Kaggle's error report.
    """
    try:
        opt = ort.SessionOptions()
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(blob, opt, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        return f"load: {str(exc)[:70]}"
    inp = sess.get_inputs()[0]
    if inp.type != "tensor(float)":
        return f"input dtype {inp.type} (grader feeds float32)"
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    x[0, 0] = 1.0
    try:
        out = sess.run(["output"], {inp.name: x})[0]
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        msg = msg.split("Status Message:")[-1].strip() if "Status" in msg else msg
        return f"infer: {msg[:70]}"
    if out.ndim != 4 or out.shape[1] != 10:
        return f"output shape {out.shape} (need (1,10,H,W))"
    if out.dtype.kind != "f":
        return f"output dtype {out.dtype} (grader thresholds a float)"
    return ""


def main() -> None:
    os.chdir(WORKING)
    names = sorted(glob.glob("task*.onnx"))
    if len(names) != 400:
        sys.exit(f"Expected 400 task onnx files, found {len(names)}; aborting.")

    # Inference gate BEFORE zipping: one poisoned model errors the whole zip.
    poisoned = []
    embedded = 0
    serialized = {}
    for name in names:
        model = onnx.load(name)  # resolves external .data sidecars
        had_external = any(
            init.data_location == onnx.TensorProto.EXTERNAL
            for init in onnx.load(name, load_external_data=False).graph.initializer
        )
        if had_external:
            embedded += 1
        blob = model.SerializeToString()
        serialized[name] = blob
        err = inference_check(blob)
        if err:
            poisoned.append((name, err))

    if poisoned:
        print(f"ABORT: {len(poisoned)} model(s) would poison the submission:")
        for name, err in poisoned:
            print("  ", name, "->", err)
        sys.exit("Refusing to build a zip that the grader will reject.")

    with zipfile.ZipFile(ZIP_PATH + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, serialized[name])
    os.replace(ZIP_PATH + ".tmp", ZIP_PATH)

    # Sanity pass: every zipped model must load standalone.
    bad = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for name in names:
            try:
                onnx.checker.check_model(onnx.load_model_from_string(zf.read(name)))
            except Exception as exc:  # noqa: BLE001 - report and continue
                bad.append((name, str(exc)[:80]))
    print(f"zipped {len(names)} models, embedded external data for {embedded}")
    print(f"inference gate: all {len(names)} models run clean at batch=1")
    print(f"standalone check: {len(names) - len(bad)} ok, {len(bad)} bad")
    for name, err in bad:
        print(" ", name, err)


if __name__ == "__main__":
    main()
