# NeuroGolf Restore Notes

This directory is the shared workspace for the Kaggle `neurogolf-2026` project.

## Restore on a New Machine

Install Python dependencies:

```bash
python -m pip install -r neurogolf/requirements.txt
```

Download the competition data once into the shared raw-data directory:

```bash
mkdir -p neurogolf/data/raw
kaggle competitions download -c neurogolf-2026 -p neurogolf/data/raw
unzip -n neurogolf/data/raw/neurogolf-2026.zip -d neurogolf/data/raw
```

The raw task JSON files and zip are intentionally ignored by git. Keep all shared data under:

- `neurogolf/data/raw`
- `neurogolf/data/working`

Private AI workspaces should store code and notes only.

## Verify the First Solved Task

Generate `task001.onnx`:

```bash
python neurogolf_codex/tools/make_task001.py
```

Then run the official helper from `neurogolf/data/raw/neurogolf_utils/neurogolf_utils.py` to verify it against task 001. The current known result is:

```text
ARC-AGI examples: 6 pass, 0 fail
ARC-GEN examples: 262 pass, 0 fail
Approximate score: 12.181 points
```
