"""Pull a public Kaggle dataset bundle and flatten its taskNNN.onnx files.

Usage: python3 bundle_pull.py <owner/dataset-slug>
Downloads via the kaggle CLI into neurogolf/data/bundles/_dl/<slug>/, then
copies every task*.onnx found (recursively, any subdir) into the flat layout
neurogolf/data/bundles/<slug>/taskNNN.onnx expected by audit_bundle.py.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

BUNDLES = Path("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/bundles")


def main():
    ref = sys.argv[1]
    slug = ref.split("/")[-1]
    dl = BUNDLES / "_dl" / slug
    out = BUNDLES / slug
    dl.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    subprocess.run(["kaggle", "datasets", "download", "-d", ref, "-p", str(dl), "--unzip", "--force"],
                   check=True)

    found = sorted(dl.rglob("task*.onnx"))
    copied = 0
    for f in found:
        m = re.match(r"task(\d+)\.onnx$", f.name)
        if not m:
            continue
        canon = out / f"task{int(m.group(1)):03d}.onnx"
        shutil.copy2(f, canon)
        copied += 1
    # keep any provenance/readme files for the audit trail
    for extra in list(dl.rglob("*.csv")) + list(dl.rglob("README*")):
        shutil.copy2(extra, out / extra.name)
    print(f"{ref}: {copied} task onnx flattened into {out}")
    if copied < 100:
        print("WARNING: fewer than 100 task files found — check the dataset layout!")


if __name__ == "__main__":
    main()
