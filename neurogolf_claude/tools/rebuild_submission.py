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

import onnx

WORKING = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
ZIP_PATH = os.path.join(WORKING, "submission.zip")


def main() -> None:
    os.chdir(WORKING)
    names = sorted(glob.glob("task*.onnx"))
    if len(names) != 400:
        sys.exit(f"Expected 400 task onnx files, found {len(names)}; aborting.")

    embedded = 0
    with zipfile.ZipFile(ZIP_PATH + ".tmp", "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            model = onnx.load(name)  # resolves external .data sidecars
            had_external = any(
                init.data_location == onnx.TensorProto.EXTERNAL
                for init in onnx.load(name, load_external_data=False).graph.initializer
            )
            if had_external:
                embedded += 1
            zf.writestr(name, model.SerializeToString())
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
    print(f"standalone check: {len(names) - len(bad)} ok, {len(bad)} bad")
    for name, err in bad:
        print(" ", name, err)


if __name__ == "__main__":
    main()
