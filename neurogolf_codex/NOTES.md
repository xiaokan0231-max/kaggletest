# Codex NeuroGolf Notes

## 2026-06-11 — Data Download + First Solved ONNX

Shared data was downloaded once into:

- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf-2026.zip`
- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task001.json` ... `task400.json`
- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils/neurogolf_utils.py`

Local Python dependencies installed in the current environment:

- `onnx`
- `onnxruntime`
- `onnx-tool`
- `ipython`
- `matplotlib`

### Solved: task001

Output rule:

For a 3x3 input grid, produce a 9x9 block grid. Each non-zero input cell activates one 3x3 block containing the full original input pattern; zero input cells produce a 3x3 zero-color block.

Generated model:

- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_codex/solutions/task001.onnx`
- Generator: `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_codex/tools/make_task001.py`

Official local verification:

```text
Results on ARC-AGI examples: 6 pass, 0 fail
Results on ARC-GEN examples: 262 pass, 0 fail
Your network IS READY for submission.
It appears to require 324000 bytes + 45005 params, yielding 12.181 points.
```

Implementation note:

The model is not size-optimal. It uses flatten + constant-index Gather + mask arithmetic. This is acceptable as a first correctness proof and should later be compressed.

### Attempted: task003

Initial visible examples suggested a 6x3 to 9x3 vertical extension plus recolor `1 -> 2`. A fixed row-map/gather model failed ARC-GEN, so the rule is not a simple periodic or fixed-row append. Do not submit the current `task003.onnx`; treat it as a failed diagnostic until the transformation is re-derived.

## 2026-06-11 — Deployment Guard

Added `neurogolf_codex/tools/deploy_guard.py` after forum #47 exposed low-score overwrite regressions for solved tasks.

Purpose:

- Refuse deployment when a candidate score is below the manifest's recorded best score unless `--allow-regression` is explicit.
- Archive the previous deployed `.onnx` before replacement.
- Reject `.onnx.data` sidecars by default so deployed models are single-file unless explicitly overridden.

Verification:

```powershell
py -3.11 -m unittest neurogolf_codex.tests.test_deploy_guard
```
