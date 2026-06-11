# NeuroGolf 2026 — Project Canon

> Created: 2026-06-11
> Competition: Kaggle `neurogolf-2026`
> Metric direction: higher is better

## Goal

Build correct and compact ONNX networks for the 400 NeuroGolf ARC-style tasks. The near-term goal is a working local evaluator, a task taxonomy, and a repeatable pipeline for generating, validating, shrinking, and submitting `taskXXX.onnx` solutions.

## Shared Data Policy

Raw Kaggle data belongs in this shared workspace only:

- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw`
- `/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working`

Private AI workspaces (`neurogolf_codex`, `neurogolf_gemini`, `neurogolf_claude`) should store experiment code, notes, generated candidates, and symlinks or config pointers to shared data. Do not duplicate the downloaded raw competition data in private workspaces.

## Initial Work Plan

1. Download the competition files once into `neurogolf/data/raw`.
2. Build a local evaluator around `neurogolf_utils.py`.
3. Classify the 400 tasks into transformation families.
4. Create minimal ONNX generators for easy task families first.
5. Record every solved family, failed assumption, and compression trick in forum topics and resolve conclusions.

## Working Rules

NeuroGolf artifacts are coordinated through the central AI Hub, not through ad-hoc file copies or git commits.

- Git stores code only: generators, reference implementations, tools, notes, and reports. Generated `taskXXX.onnx` files and `submission.zip` do not go through git.
- The central artifact store is `neurogolf/data/working` on the Hub machine. All deployable ONNX files must be published through the Hub deploy API.
- Do not overwrite `data/working/taskXXX.onnx` by hand. A candidate must pass official verification, be non-dummy, and beat the recorded best score unless an explicit regression override is documented.
- Every deployed model records metadata: task id, score, official verification status, SHA-256, byte size, source forum topic, creator, deployed path, dummy flag, and timestamp.
- Replaced models must be archived under `neurogolf/data/working/archive/` with task id, score, source topic, and SHA-256 in the filename.
- Every successful deploy rebuilds `submission.zip` and verifies that the zip-contained ONNX models can be loaded independently.
- Forum conclusion status is not task completion. A task counts as solved only when Hub artifact metadata says `verified_status == IS_READY`, `is_deployed == true`, and `is_dummy == false`.
- Failed or postponed investigations must still be resolved in the forum, but they do not count as solved tasks. For example, a rule-discovery closure like `task003` is a knowledge result, not an ONNX solve.
- Kaggle credentials never go into git or MySQL. Prefer downloading the Hub-built `submission.zip` and submitting locally; center-machine submission may be enabled only by an explicit environment switch.

Remote submission workflow from any client:

```powershell
py -3.11 tools/pull_submission.py --hub http://192.168.137.215:8000 --out neurogolf/data/working/submission.zip
kaggle competitions submit -c neurogolf-2026 -f neurogolf/data/working/submission.zip -m "NeuroGolf Hub submission"
```

Remote artifact deployment workflow from any client:

```powershell
py -3.11 tools/deploy_neurogolf_artifact.py --hub http://192.168.137.215:8000 --task task015 --model path/to/task015.onnx --score 13.080 --topic 28 --agent Gemini
```

## First Questions

- Which ONNX ops are allowed and sufficient for the common ARC transformations?
- Which tasks can be solved with constant/grid-copy/color-map logic before any neural search?
- How should we score candidate solutions locally before spending Kaggle submissions?
