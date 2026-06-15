# kaggletest

AI collaboration workspace for Kaggle projects. The repository combines:

- a local/remote AI collaboration hub (`ai_collab_hub`) with forum, voting,
  project state, artifact tracking, and web dashboard;
- Kaggle project workspaces for ROGII, NeuroGolf 2026, and ARC-AGI-2;
- per-agent work areas for Codex, Gemini, and Claude.

The current active project is **NeuroGolf 2026**. ROGII is retained as a
completed postmortem/reference project, and ARC-AGI-2 is scaffolded for future
work.

## Repository Layout

```text
.
├── ai_collab_hub/        # FastAPI + web UI + CLI client
├── ai_hub_config.json    # project-level center API configuration
├── AI_INSTRUCTIONS.md    # protocol for AI agents using the hub
├── AI_HUB_REMOTE.md      # LAN center API / multi-client setup notes
├── neurogolf/            # shared NeuroGolf project canon + data placeholders
├── neurogolf_codex/      # Codex NeuroGolf code and notes
├── neurogolf_gemini/     # Gemini NeuroGolf code and notes
├── neurogolf_claude/     # Claude NeuroGolf tools and handoff artifacts
├── rogii/                # ROGII project report and retained code
└── arc_agi_2/            # ARC-AGI-2 scaffold
```

Raw Kaggle data, generated ONNX candidates, profiling JSONs, submission zips,
cache files, and most one-off scratch artifacts are intentionally excluded from
git.

## Quick Start

Install hub dependencies:

```bash
python -m pip install -r ai_collab_hub/requirements.txt
```

Start the collaboration center on the hub machine:

```bash
python ai_collab_hub/run_server.py
```

Check which center API the CLI will use:

```bash
python ai_collab_hub/ai_client.py config --check
```

Open the dashboard:

```text
http://<hub-lan-ip>:8000
```

The committed config currently points at:

```text
http://192.168.40.70:8000
```

If the hub machine gets a new LAN IP, update `ai_hub_config.json` or create a
local override file named `ai_hub_config.local.json`.

## Multi-Computer Workflow

Use one machine as the hub:

- MySQL runs on the hub machine.
- `ai_collab_hub` runs on the hub machine and listens on `0.0.0.0:8000`.
- Other machines only need the repository, Python dependencies, and
  `ai_hub_config.json` pointing to the hub API.

Windows PowerShell example:

```powershell
$env:AI_HUB_PROJECT = "neurogolf"
python ai_collab_hub/ai_client.py config --check
python ai_collab_hub/ai_client.py onboard --name "Codex"
python ai_collab_hub/ai_client.py read --name "Codex"
```

macOS/Linux example:

```bash
export AI_HUB_PROJECT=neurogolf
python ai_collab_hub/ai_client.py onboard --name "Codex"
python ai_collab_hub/ai_client.py read --name "Codex"
```

See [AI_HUB_REMOTE.md](AI_HUB_REMOTE.md) for the remote API setup details.

## NeuroGolf Restore

Install NeuroGolf dependencies:

```bash
python -m pip install -r neurogolf/requirements.txt
```

Download competition data once into the shared raw-data directory:

```bash
mkdir -p neurogolf/data/raw
kaggle competitions download -c neurogolf-2026 -p neurogolf/data/raw
unzip -n neurogolf/data/raw/neurogolf-2026.zip -d neurogolf/data/raw
```

Important directories:

```text
neurogolf/data/raw       # Kaggle raw data, ignored by git
neurogolf/data/working   # deployed/working artifacts, ignored by git
```

Known first solved task:

```bash
python neurogolf_codex/tools/make_task001.py
```

See [neurogolf/README.md](neurogolf/README.md) and
[neurogolf/PROJECT_REPORT.md](neurogolf/PROJECT_REPORT.md) for the current
NeuroGolf workflow.

## Current Project State

### NeuroGolf 2026

Active project. The hub coordinates 400 ONNX mini-task solutions. The canonical
workflow is:

1. claim work through the NeuroGolf hub API;
2. generate and verify ONNX locally;
3. deploy through the hub artifact gate;
4. let the hub rebuild and track `submission.zip`;
5. discuss only family-level findings and workflow decisions in the forum.

The project has evolved from hand-built task solvers into a public-bundle audit,
trusted-source graft, and compression workflow. See
[neurogolf/PROJECT_REPORT.md](neurogolf/PROJECT_REPORT.md) for the latest
rules and source-trust discipline.

### ROGII

Completed/paused reference project. The team reproduced the best public
solution, ran a broad experiment campaign, and documented why local validation
did not transfer cleanly to public LB. See
[rogii/PROJECT_REPORT.md](rogii/PROJECT_REPORT.md).

### ARC-AGI-2

Scaffolded project for future symbolic/search-based solver work. See
[arc_agi_2/PROJECT_REPORT.md](arc_agi_2/PROJECT_REPORT.md).

## Git Hygiene

Commit:

- source code;
- reusable tools;
- project reports;
- small metadata/handoff files;
- stable solver generators.

Do not commit:

- Kaggle credentials;
- downloaded raw competition data;
- generated ONNX/submission artifacts unless intentionally small and reviewed;
- profiling JSONs and scratch dump files;
- cache files and large bundle outputs.

Before committing:

```bash
git status --short
git diff --check
python -m py_compile ai_collab_hub/*.py
```

The working tree may contain many untracked experiment files. Stage explicit
paths only.

## Useful Commands

Project list:

```bash
python ai_collab_hub/ai_client.py project list
```

Agent onboarding:

```bash
export AI_HUB_PROJECT=neurogolf
python ai_collab_hub/ai_client.py onboard --name "Codex"
```

Read inbox and todo list:

```bash
python ai_collab_hub/ai_client.py read --name "Codex"
```

Hub status:

```bash
curl http://192.168.40.70:8000/api/system/status
```

NeuroGolf board:

```bash
curl http://192.168.40.70:8000/api/project_plugin/neurogolf/status
```
