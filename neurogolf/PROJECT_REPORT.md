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

## First Questions

- Which ONNX ops are allowed and sufficient for the common ARC transformations?
- Which tasks can be solved with constant/grid-copy/color-map logic before any neural search?
- How should we score candidate solutions locally before spending Kaggle submissions?
