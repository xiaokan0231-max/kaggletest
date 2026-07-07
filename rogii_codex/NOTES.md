# Codex Experiment Notes

## 2026-06-09 — Spatial Trust Gate / v7 Direction

Goal: evaluate whether v6's spatial branch should stay fixed at EGFDU `w=0.08`,
be gated by known-prefix reliability, or use a different surface/weight.

All experiments read shared inputs from `../rogii` only:

- `../rogii/data/extracted/train`
- `../rogii/selcache.pkl`
- `../rogii/train_side.pkl`

No `../rogii` files were modified.

### Tooling

`python -m rogii_codex.spatial_gate`

What it does:

- builds leave-one-well IDW surface interpolation from train wells;
- calibrates each surface on known `TVT_input`;
- evaluates selector-only, spatial-only, fixed selector/spatial blend, and gated blend;
- when `train_side.pkl` is present, also evaluates final `0.3*GBDT + 0.7*selector_branch`.

### Full-Train Results

Settings:

- `stride=15`
- `k=12`
- 770 evaluated wells; 3 wells skipped because the first neighbor pool was all self-well after leave-one exclusion.

#### EGFDU Only

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m rogii_codex.spatial_gate \
  --n 0 --seed 0 --mode egfdu --stride 15 --fixed-weight 0.08 \
  --max-weight 0.12 --sweep --save
```

Selector branch:

| spatial weight | RMSE |
|---:|---:|
| 0.00 | 10.5035 |
| 0.08 | 10.0261 |
| 0.12 | 9.9367 |
| 0.14 | 9.9310 |

Final blend (`0.3*GBDT + 0.7*selector_branch`):

| spatial weight | RMSE |
|---:|---:|
| 0.00 | 9.5618 |
| 0.08 | 9.3513 |
| 0.10 | 9.3321 |
| 0.12 | 9.3265 |
| 0.14 | 9.3345 |

Takeaway: v6's `w=0.08` is directionally right and captures most of the gain,
but full-train final blend prefers `w≈0.12`.

#### Best Surface by Known-Tail RMSE

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m rogii_codex.spatial_gate \
  --n 0 --seed 0 --mode best --stride 15 --fixed-weight 0.12 \
  --max-weight 0.12 --sweep --save
```

Selector branch:

| spatial weight | RMSE |
|---:|---:|
| 0.00 | 10.5035 |
| 0.08 | 10.0079 |
| 0.12 | 9.9043 |
| 0.14 | 9.8904 |

Final blend:

| spatial weight | RMSE |
|---:|---:|
| 0.00 | 9.5618 |
| 0.08 | 9.3395 |
| 0.10 | 9.3163 |
| 0.12 | 9.3063 |
| 0.14 | 9.3095 |

Best-surface counts:

| surface | wells |
|---|---:|
| EGFDL | 151 |
| ANCC | 132 |
| ASTNU | 127 |
| BUDA | 125 |
| ASTNL | 123 |
| EGFDU | 112 |

Takeaway: choosing the surface with the best known-tail fit beats fixed EGFDU
on full-train evaluation. The margin is small but real in this setup:
`9.3063` vs `9.3265` final RMSE at `w=0.12`.

### Gate Attempt

The reliability gate used:

```text
w = max_weight
    * 1 / (1 + (best_tail_rmse / err_scale)^2)
    * 1 / (1 + (median_dist / dist_scale)^2)
```

With `max_weight=0.12`, `err_scale=5`, `dist_scale=500`.

Full-train final blend:

| branch | RMSE |
|---|---:|
| final selector | 9.5618 |
| EGFDU fixed `w=0.08` | 9.3513 |
| EGFDU gated | 9.4146 |
| best-surface fixed `w=0.12` | 9.3063 |
| best-surface gated | 9.4117 |

Takeaway: this gate is too conservative. It improves more individual wells, but
it misses large pooled-RMSE wins where selector is badly wrong. Known-prefix
error and distance weakly predict spatial harm, but not enough to beat a fixed
small spatial weight.

### Current v7 Candidate

Best candidate from Codex line:

```text
mode = best surface by known-tail RMSE
selector-branch spatial weight = 0.12
final GBDT weight unchanged = 0.3
```

Expected local effect:

```text
final blend: 9.5618 -> 9.3063
```

Compared with current v6 local analogue:

```text
EGFDU fixed w=0.08: 9.3513
best-surface fixed w=0.12: 9.3063
```

This is a small but plausible improvement over v6. Because public LB noise and
distribution shift are large, v6 should finish scoring before deciding whether
to submit v7.

## 2026-06-09 — No-Op-Safe Fingerprint Matcher Audit

Goal: check whether there is a low-risk exact/near-exact well matcher that can
copy train `TVT` only when a test well is essentially identical to a train well.

Tool:

```bash
python -m rogii_codex.fingerprint_match
```

Trigger policy:

- same row count;
- same `TVT_input` prefix length;
- full observable arrays `MD/X/Y/Z/GR` match within strict max-abs thresholds;
- known-prefix `TVT_input` matches within strict max-abs threshold;
- train-only columns such as `TVT` and formation surfaces are never used for
  matching.

### Visible Test Audit

Including same ID:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m rogii_codex.fingerprint_match \
  --mode all --top 3 --save
```

Result:

| test well | chosen train well | triggered | copy RMSE |
|---|---|---:|---:|
| 000d7d20 | 000d7d20 | true | 0.000000 |
| 00bbac68 | 00bbac68 | true | 0.000000 |
| 00e12e8b | 00e12e8b | true | 0.000000 |

Excluding same ID:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m rogii_codex.fingerprint_match \
  --mode test --top 3 --exclude-same-id --save
```

Result: no triggers for all 3 visible test wells.

Train duplicate observable fingerprints:

```text
duplicate groups = 0
duplicate wells = 0 / 773
```

Takeaway:

- The matcher correctly recovers the known same-ID leakage case.
- It does not find any non-ID exact duplicate in local visible test or train.
- A no-op-safe notebook hook is possible, but current local evidence says it
  would probably do nothing unless hidden test contains non-ID exact/near-exact
  duplicates.
- This is safer than spatial, but likely low expected value.
