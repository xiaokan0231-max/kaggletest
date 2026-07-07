from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import default_paths


PATHS = default_paths()
if str(PATHS.rogii_dir) not in sys.path:
    sys.path.insert(0, str(PATHS.rogii_dir))

import heur  # noqa: E402


CANDIDATES = ("selector", "pf_mean", "mix_pf_mean_0.12", "beam", "const")


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.astype(float) - truth.astype(float)) ** 2)))


def load_selector_cache() -> list[dict]:
    cache_path = PATHS.rogii_dir / "selcache2.pkl"
    if not cache_path.exists():
        cache_path = PATHS.rogii_dir / "selcache.pkl"
    with cache_path.open("rb") as f:
        return pickle.load(f)


def well_paths(wid: str) -> tuple[Path, Path]:
    return (
        PATHS.train_dir / f"{wid}__horizontal_well.csv",
        PATHS.train_dir / f"{wid}__typewell.csv",
    )


def load_train_well(wid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hw_path, tw_path = well_paths(wid)
    return pd.read_csv(hw_path), pd.read_csv(tw_path)


def known_prefix_len(hw: pd.DataFrame) -> int:
    return int(hw["TVT_input"].notna().sum())


def const_full(hw: pd.DataFrame) -> np.ndarray:
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    ps = int(np.sum(~np.isnan(tvt_input)))
    last = float(tvt_input[ps - 1])
    out = tvt_input.copy()
    out[ps:] = last
    return out


def candidate_full_predictions(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    n_seeds: int,
) -> dict[str, np.ndarray]:
    pf_by_scale = heur.run_pf_lik_ensemble_scales(hw, tw, n_seeds=n_seeds)
    beam = heur.run_beam_ensemble(hw, tw)
    variant = heur.selector_well_code(hw)[1]
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    ps = int(np.sum(~np.isnan(tvt_input)))
    last = float(tvt_input[ps - 1])
    selector = heur.apply_selector_variant(variant, pf_by_scale, beam, last)
    pf_mean = pf_by_scale["pf_mean"]
    return {
        "selector": selector,
        "pf_mean": pf_mean,
        "mix_pf_mean_0.12": 0.88 * selector + 0.12 * pf_mean,
        "beam": beam,
        "const": const_full(hw),
    }


def actual_eval_candidates(entry: dict) -> dict[str, np.ndarray]:
    selector = np.asarray(entry["pred"], dtype=float)
    pf_mean = np.asarray(entry.get("pf_mean", selector), dtype=float)
    beam = np.asarray(entry.get("beam", selector), dtype=float)
    const = np.full_like(selector, float(entry["last"]), dtype=float)
    return {
        "selector": selector,
        "pf_mean": pf_mean,
        "mix_pf_mean_0.12": 0.88 * selector + 0.12 * pf_mean,
        "beam": beam,
        "const": const,
    }


def pseudo_tail_scores(
    wid: str,
    cuts: list[int],
    n_seeds: int,
    min_prefix: int,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    hw, tw = load_train_well(wid)
    actual_ps = known_prefix_len(hw)
    scores = {name: [] for name in CANDIDATES}
    wins: Counter[str] = Counter()

    for cut in cuts:
        pseudo_ps = actual_ps - cut
        if pseudo_ps < min_prefix or cut < 5:
            continue
        hw_cut = hw.copy()
        hw_cut.loc[pseudo_ps:, "TVT_input"] = np.nan
        preds = candidate_full_predictions(hw_cut, tw, n_seeds=n_seeds)
        truth = hw["TVT"].to_numpy(dtype=float)[pseudo_ps:actual_ps]
        cut_scores = {}
        for name in CANDIDATES:
            cut_scores[name] = rmse(preds[name][pseudo_ps:actual_ps], truth)
            scores[name].append(cut_scores[name])
        wins[min(cut_scores, key=cut_scores.get)] += 1
    return scores, dict(wins)


def choose_candidate(
    avg_scores: dict[str, float],
    wins: dict[str, int],
    margin: float,
    min_wins: int,
    default: str = "selector",
) -> tuple[str, str]:
    valid = {k: v for k, v in avg_scores.items() if np.isfinite(v)}
    if default not in valid or len(valid) <= 1:
        return default, "insufficient_scores"
    best = min(valid, key=valid.get)
    if best == default:
        return default, "default_best"
    if valid[default] - valid[best] < margin:
        return default, "margin"
    if int(wins.get(best, 0)) < min_wins:
        return default, "consistency"
    return best, "trigger"


def evaluate_entries(
    entries: list[dict],
    cuts: list[int],
    n_seeds: int,
    min_prefix: int,
    margins: list[float],
    min_wins: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    t0 = time.time()
    for i, entry in enumerate(entries, 1):
        wid = entry["wid"]
        try:
            scores, wins = pseudo_tail_scores(wid, cuts, n_seeds, min_prefix)
        except Exception as exc:
            rows.append({"wid": wid, "error": repr(exc)})
            continue

        avg_scores = {
            name: float(np.mean(vals)) if vals else float("inf")
            for name, vals in scores.items()
        }
        actual = actual_eval_candidates(entry)
        truth = np.asarray(entry["truth"], dtype=float)
        actual_rmse = {name: rmse(pred, truth) for name, pred in actual.items()}

        base = {
            "wid": wid,
            "ps": int(entry.get("ps", -1)),
            "eval_len": int(len(truth)),
            "n_cutpoints": int(len(scores["selector"])),
            "wins": json.dumps(wins, sort_keys=True),
            "self_selector": avg_scores["selector"],
            "self_pf_mean": avg_scores["pf_mean"],
            "self_mix_pf_mean_0.12": avg_scores["mix_pf_mean_0.12"],
            "self_beam": avg_scores["beam"],
            "self_const": avg_scores["const"],
            "eval_selector": actual_rmse["selector"],
            "eval_pf_mean": actual_rmse["pf_mean"],
            "eval_mix_pf_mean_0.12": actual_rmse["mix_pf_mean_0.12"],
            "eval_beam": actual_rmse["beam"],
            "eval_const": actual_rmse["const"],
        }
        for margin in margins:
            chosen, reason = choose_candidate(avg_scores, wins, margin, min_wins)
            base[f"chosen_m{margin:g}"] = chosen
            base[f"reason_m{margin:g}"] = reason
            base[f"eval_chosen_m{margin:g}"] = actual_rmse[chosen]
        rows.append(base)
        if i % 10 == 0:
            print(f"processed {i}/{len(entries)} wells in {time.time() - t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    summary = summarize(df, margins)
    summary["n_seeds"] = n_seeds
    summary["cuts"] = cuts
    summary["min_prefix"] = min_prefix
    summary["min_wins"] = min_wins
    summary["elapsed_sec"] = time.time() - t0
    return df, summary


def pooled_rmse_from_rows(df: pd.DataFrame, col: str) -> float:
    ok = df[col].notna() & df["eval_len"].notna()
    if not ok.any():
        return float("nan")
    weights = df.loc[ok, "eval_len"].astype(float).to_numpy()
    vals = df.loc[ok, col].astype(float).to_numpy()
    return float(np.sqrt(np.sum(weights * vals**2) / np.sum(weights)))


def summarize(df: pd.DataFrame, margins: list[float]) -> dict:
    out: dict[str, object] = {
        "wells": int(len(df)),
        "usable_wells": int(df.get("n_cutpoints", pd.Series(dtype=float)).fillna(0).gt(0).sum()),
        "selector_pooled_rmse": pooled_rmse_from_rows(df, "eval_selector"),
        "candidate_eval_pooled_rmse": {
            name: pooled_rmse_from_rows(df, f"eval_{name}")
            for name in CANDIDATES
        },
    }
    margin_results = {}
    for margin in margins:
        chosen_col = f"chosen_m{margin:g}"
        eval_col = f"eval_chosen_m{margin:g}"
        if chosen_col not in df:
            continue
        triggered = df[chosen_col].fillna("selector") != "selector"
        margin_results[str(margin)] = {
            "pooled_rmse": pooled_rmse_from_rows(df, eval_col),
            "triggered_wells": int(triggered.sum()),
            "trigger_rate": float(triggered.mean()) if len(df) else 0.0,
            "chosen_counts": df[chosen_col].fillna("error").value_counts().to_dict(),
            "reason_counts": df[f"reason_m{margin:g}"].fillna("error").value_counts().to_dict(),
        }
    out["gate_by_margin"] = margin_results
    return out


def select_entries(cache: list[dict], n: int, seed: int, min_eval: int) -> list[dict]:
    entries = [e for e in cache if len(e.get("truth", [])) >= min_eval]
    if n and n < len(entries):
        rng = random.Random(seed)
        entries = rng.sample(entries, n)
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=80, help="sampled wells; 0 means all usable wells")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=16, help="PF seeds per pseudo cutpoint")
    parser.add_argument("--cuts", default="40,80,120")
    parser.add_argument("--min-prefix", type=int, default=80)
    parser.add_argument("--min-eval", type=int, default=30)
    parser.add_argument("--margins", default="0,0.25,0.5,1.0,2.0")
    parser.add_argument("--min-wins", type=int, default=2)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cuts = [int(x) for x in args.cuts.split(",") if x]
    margins = [float(x) for x in args.margins.split(",") if x]
    cache = load_selector_cache()
    entries = select_entries(cache, args.n, args.seed, args.min_eval)
    df, summary = evaluate_entries(
        entries=entries,
        cuts=cuts,
        n_seeds=args.n_seeds,
        min_prefix=args.min_prefix,
        margins=margins,
        min_wins=args.min_wins,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.save:
        PATHS.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"prefix_gate_n{len(entries)}_seed{args.seed}_pf{args.n_seeds}"
        csv_path = PATHS.output_dir / f"{stem}.csv"
        json_path = PATHS.output_dir / f"{stem}.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"saved {csv_path}")
        print(f"saved {json_path}")


if __name__ == "__main__":
    main()
