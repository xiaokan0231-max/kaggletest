# rogii_codex

Codex-side experiments for the ROGII Wellbore Geology Prediction project.

Boundary rule:

- `../rogii` is treated as read-only shared state and Claude's line of work.
- New Codex code, notes, and outputs go under `rogii_codex/`.
- Scripts may read `../rogii/data`, `../rogii/artifacts`, and caches such as
  `../rogii/selcache.pkl`, but should not write into `../rogii`.

Current first direction: v7 spatial trust gate.

The existing v6 mixes an EGFDU spatial reconstruction into the selector branch
with a fixed weight. The next test is to make that spatial weight per-well and
quality-gated using only the known TVT_input segment:

- build spatial surface predictions from neighboring train wells;
- estimate each surface's offset on the known prefix;
- score reliability by known-prefix error, tail error, and neighbor distance;
- blend spatial into selector only when the known prefix says it is trustworthy.

Useful commands:

```bash
python -m rogii_codex.spatial_gate --n 80 --mode weighted
python -m rogii_codex.spatial_gate --n 160 --mode egfdu
```

