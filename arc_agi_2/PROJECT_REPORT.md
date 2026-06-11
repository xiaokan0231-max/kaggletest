# ARC-AGI-2 — Project Canon

> Created: 2026-06-11
> Competition: Kaggle `arc-prize-2026-arc-agi-2`
> Metric direction: higher is better

## Goal

Build a locally verifiable ARC-AGI-2 solver stack based on task taxonomy, symbolic rules, search, program synthesis, and optional LLM assistance. The project should prioritize reusable reasoning infrastructure and durable negative results over leaderboard noise.

## Shared Data Policy

Raw Kaggle data belongs in this shared workspace only:

- `/Users/kanxiao/IdeaProjects/kaggletest/arc_agi_2/data/raw`
- `/Users/kanxiao/IdeaProjects/kaggletest/arc_agi_2/data/working`

Private AI workspaces (`arc_agi_2_codex`, `arc_agi_2_gemini`, `arc_agi_2_claude`) should store experiment code, private notes, solver variants, and symlinks or config pointers to shared data. Do not duplicate the downloaded raw competition data in private workspaces.

## Initial Work Plan

1. Download competition JSON files once into `arc_agi_2/data/raw`.
2. Build a local evaluator matching Kaggle submission format.
3. Create a task taxonomy and baseline solver registry.
4. Add symbolic primitives and search strategies with strict train/eval separation.
5. Use forum topics for every proposed solver family, with explicit evidence and resolve conclusions.

## First Questions

- Which public training tasks are solved by simple object/color/geometry primitives?
- Which failures are due to missing primitives versus search explosion?
- How can LLMs help propose programs without contaminating evaluation discipline?
