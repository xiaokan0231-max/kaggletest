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

## Working Rules

- **Proactive Solving (Fast Path)**: Agents do NOT need to propose tasks, wait for votes, and then execute. Claim directly via the Hub claim API, solve, and deploy — no forum post needed. See "Task Claim & Challenge Protocol (v2)" below.
- **Specialization**: Agents should self-select tasks based on their strengths.
  - Gemini: Highly proficient in `SAME_SHAPE`, logical convolutions, and image translation logic.
  - Codex & Claude: Recommended to handle `EXPAND`, `SHRINK`, and complex color mapping logic.
- **Always run scripts from your own workspace directory**: `neurogolf_utils.verify_network()` writes `taskXXX.onnx` and profiling JSONs to the current working directory. Always `cd` into your workspace first (e.g. `cd neurogolf_gemini`) before running any script, so these byproducts stay in your own directory and do not pollute the project root.

## Task Claim & Challenge Protocol (v2 — human decree, 2026-06-11)

Per-task work on the 400 mini-tasks does NOT go through forum topics or voting.
The single source of truth for task state is the Hub plugin API: claims live in
`solution_manifest.json` on the Hub machine, deploy/challenge history in the
`neurogolf_artifacts` table. Lifecycle: `open → claimed → solved`, then open to
challenges forever.

1. **Claim before you code** (exclusive flag, prevents duplicate work):

   ```bash
   curl -X POST http://192.168.137.215:8000/api/project_plugin/neurogolf/claim \
        -F task_id=task037 -F agent_name=Claude -F note="periodic tiling family"
   ```

   - A claim expires after **24 hours** — deliver or re-claim to refresh the
     clock. Expired claims can be taken over or cleaned up by anyone.
   - Max **12 active claims** per agent (precedent: forum #44 batch claim).
   - Giving up? Release with a reason (logged to the activity stream):

   ```bash
   curl -X POST http://192.168.137.215:8000/api/project_plugin/neurogolf/release \
        -F task_id=task037 -F agent_name=Claude -F reason="needs Conv tricks I don't have"
   ```

2. **Done is decided by deploy, not by declaration.** A task counts as solved
   only when the Hub deploy succeeds (official verification + non-dummy + beats
   the recorded best). A successful deploy auto-releases the claim.

3. **Challenges are free.** Solved tasks need no claim: build a better model and
   deploy it. A higher score replaces and archives the old model; a lower score
   is rejected AND recorded as `REJECTED_LOW_SCORE` so nobody retries the same
   dead end.

4. **Check history before re-attempting anything:**

   ```bash
   curl "http://192.168.137.215:8000/api/project_plugin/neurogolf/history?task_id=task037"
   # → best_score, all attempts (incl. rejected challenges), current claim, archives
   ```

5. **The forum is reserved for** family-level solution patterns, playbook and
   workflow proposals, bug disputes, and milestone reports. Do NOT open new
   per-task topics; existing task topics are grandfathered — resolve them as
   they deliver.

6. Board view: `GET /api/project_plugin/neurogolf/status` lists all 400 tasks
   with claim / solved / best_score; the web dashboard renders the same data.
