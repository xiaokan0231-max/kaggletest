from __future__ import annotations

import argparse
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import default_paths


SURFACES = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


@dataclass
class SurfaceSamples:
    xy: np.ndarray
    surfaces: np.ndarray
    wells: np.ndarray
    tree: cKDTree


@dataclass
class SpatialResult:
    pred_full: np.ndarray
    surface_weights: np.ndarray
    known_rmse: np.ndarray
    tail_rmse: np.ndarray
    median_dist: float
    gate_weight: float


def rmse(err: np.ndarray) -> float:
    err = np.asarray(err, dtype=float)
    return float(np.sqrt(np.nanmean(err * err)))


def train_well_ids(train_dir: Path) -> list[str]:
    return sorted(p.name.split("__")[0] for p in train_dir.glob("*__horizontal_well.csv"))


def load_horizontal(train_dir: Path, wid: str) -> pd.DataFrame:
    return pd.read_csv(train_dir / f"{wid}__horizontal_well.csv")


def build_surface_samples(train_dir: Path, stride: int = 15) -> SurfaceSamples:
    xy_parts: list[np.ndarray] = []
    surface_parts: list[np.ndarray] = []
    well_parts: list[np.ndarray] = []

    for wid in train_well_ids(train_dir):
        try:
            df = pd.read_csv(
                train_dir / f"{wid}__horizontal_well.csv",
                usecols=["X", "Y", *SURFACES],
            ).dropna(subset=["X", "Y", *SURFACES])
        except Exception:
            continue

        if df.empty:
            continue

        sub = df.iloc[::stride]
        xy_parts.append(sub[["X", "Y"]].to_numpy(dtype=float))
        surface_parts.append(sub[list(SURFACES)].to_numpy(dtype=float))
        well_parts.append(np.full(len(sub), wid, dtype=object))

    if not xy_parts:
        raise RuntimeError(f"No surface samples found in {train_dir}")

    xy = np.vstack(xy_parts)
    surfaces = np.vstack(surface_parts)
    wells = np.concatenate(well_parts)
    return SurfaceSamples(xy=xy, surfaces=surfaces, wells=wells, tree=cKDTree(xy))


def query_surface_idw(
    samples: SurfaceSamples,
    points: np.ndarray,
    query_well: str | None,
    k: int,
    fetch_extra: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_fetch = min(len(samples.wells), max(k, k + fetch_extra))
    dist, idx = samples.tree.query(points, k=n_fetch)
    if idx.ndim == 1:
        idx = idx[:, None]
        dist = dist[:, None]

    out = np.empty((len(points), len(SURFACES)), dtype=float)
    min_dist = np.empty(len(points), dtype=float)

    for row in range(len(points)):
        ii = idx[row]
        dd = dist[row]
        if query_well is not None:
            keep = samples.wells[ii] != query_well
            ii = ii[keep]
            dd = dd[keep]
        ii = ii[:k]
        dd = dd[:k]
        if len(ii) == 0:
            raise RuntimeError("No neighbors left after self-well exclusion")

        weights = 1.0 / (dd + 1.0)
        weights = weights / weights.sum()
        out[row] = (weights[:, None] * samples.surfaces[ii]).sum(axis=0)
        min_dist[row] = float(dd.min())

    return out, min_dist


def prefix_count(tvt_input: np.ndarray) -> int:
    return int(np.isfinite(tvt_input).sum())


def surface_predictions(
    hw: pd.DataFrame,
    surf_interp: np.ndarray,
    tail_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = hw["Z"].to_numpy(dtype=float)
    tvt_input = hw["TVT_input"].to_numpy(dtype=float)
    ps = prefix_count(tvt_input)
    if ps < 2:
        raise RuntimeError("Too few known TVT_input rows for spatial calibration")

    preds = []
    known_rmse = []
    tail_rmse = []
    tail_start = max(0, ps - tail_points)

    for surface_idx in range(len(SURFACES)):
        raw = -(z - surf_interp[:, surface_idx])
        offset = float(np.nanmean(tvt_input[:ps] - raw[:ps]))
        pred = raw + offset
        preds.append(pred)
        known_rmse.append(rmse(pred[:ps] - tvt_input[:ps]))
        tail_rmse.append(rmse(pred[tail_start:ps] - tvt_input[tail_start:ps]))

    return np.vstack(preds), np.asarray(known_rmse), np.asarray(tail_rmse)


def combine_surfaces(
    pred_by_surface: np.ndarray,
    tail_rmse: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "egfdu":
        weights = np.zeros(len(SURFACES), dtype=float)
        weights[SURFACES.index("EGFDU")] = 1.0
    elif mode == "best":
        weights = np.zeros(len(SURFACES), dtype=float)
        weights[int(np.nanargmin(tail_rmse))] = 1.0
    elif mode == "weighted":
        safe = np.maximum(tail_rmse, 1e-3)
        weights = 1.0 / (safe * safe)
        weights = weights / weights.sum()
    else:
        raise ValueError(f"Unknown surface mode: {mode}")

    pred = (weights[:, None] * pred_by_surface).sum(axis=0)
    return pred, weights


def gate_weight(
    tail_rmse: np.ndarray,
    median_dist: float,
    max_weight: float,
    err_scale: float,
    dist_scale: float,
) -> float:
    best_tail = float(np.nanmin(tail_rmse))
    err_quality = 1.0 / (1.0 + (best_tail / err_scale) ** 2)
    dist_quality = 1.0 / (1.0 + (median_dist / dist_scale) ** 2)
    return float(max_weight * err_quality * dist_quality)


def spatial_predict(
    hw: pd.DataFrame,
    samples: SurfaceSamples,
    query_well: str | None,
    k: int,
    mode: str,
    max_weight: float,
    err_scale: float,
    dist_scale: float,
    tail_points: int,
    fetch_extra: int,
) -> SpatialResult:
    points = hw[["X", "Y"]].to_numpy(dtype=float)
    surf_interp, min_dist = query_surface_idw(
        samples,
        points,
        query_well=query_well,
        k=k,
        fetch_extra=fetch_extra,
    )
    pred_by_surface, known_rmse, tail_rmse = surface_predictions(hw, surf_interp, tail_points)
    pred_full, weights = combine_surfaces(pred_by_surface, tail_rmse, mode)
    median_dist = float(np.nanmedian(min_dist))
    w_gate = gate_weight(tail_rmse, median_dist, max_weight, err_scale, dist_scale)
    return SpatialResult(
        pred_full=pred_full,
        surface_weights=weights,
        known_rmse=known_rmse,
        tail_rmse=tail_rmse,
        median_dist=median_dist,
        gate_weight=w_gate,
    )


def load_selector_cache(path: Path) -> dict[str, dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    return {row["wid"]: row for row in rows}


def load_gbdt_oof(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        side = pickle.load(f)
    df = pd.DataFrame({"well": side["well"], "gbdt": side["gbdt_oof"]})
    return {well: group["gbdt"].to_numpy(dtype=float) for well, group in df.groupby("well", sort=False)}


def evaluate(
    n: int | None,
    seed: int,
    stride: int,
    k: int,
    mode: str,
    fixed_weight: float,
    max_weight: float,
    err_scale: float,
    dist_scale: float,
    tail_points: int,
    fetch_extra: int,
    sweep: bool,
    save: bool,
) -> None:
    paths = default_paths()
    selector_cache = load_selector_cache(paths.selector_cache)
    gbdt_cache = load_gbdt_oof(paths.train_side_cache)
    all_ids = [wid for wid in train_well_ids(paths.train_dir) if wid in selector_cache]
    random.seed(seed)
    sample_ids = random.sample(all_ids, min(n or len(all_ids), len(all_ids)))

    t0 = time.time()
    print(f"Building surface samples from {paths.train_dir} stride={stride} ...", flush=True)
    samples = build_surface_samples(paths.train_dir, stride=stride)
    print(f"Surface samples: {len(samples.xy):,} points in {time.time() - t0:.1f}s", flush=True)

    err_sel = []
    err_sp = []
    err_fixed = []
    err_gate = []
    err_final_sel = []
    err_final_fixed = []
    err_final_gate = []
    rows = []
    raw_predictions = []

    for wid in sample_ids:
        hw = load_horizontal(paths.train_dir, wid)
        ps = prefix_count(hw["TVT_input"].to_numpy(dtype=float))
        cache = selector_cache[wid]
        truth = cache["truth"].astype(float)
        sel = cache["pred"].astype(float)
        gbdt = gbdt_cache.get(wid)

        try:
            sp = spatial_predict(
                hw=hw,
                samples=samples,
                query_well=wid,
                k=k,
                mode=mode,
                max_weight=max_weight,
                err_scale=err_scale,
                dist_scale=dist_scale,
                tail_points=tail_points,
                fetch_extra=fetch_extra,
            )
        except Exception as exc:
            print(f"[skip] {wid}: {exc}")
            continue

        sp_post = sp.pred_full[ps:].astype(float)
        if len(sp_post) != len(sel):
            print(f"[skip] {wid}: selector/spatial length mismatch {len(sel)} vs {len(sp_post)}")
            continue

        fixed = (1.0 - fixed_weight) * sel + fixed_weight * sp_post
        gated = (1.0 - sp.gate_weight) * sel + sp.gate_weight * sp_post

        err_sel.append(sel - truth)
        err_sp.append(sp_post - truth)
        err_fixed.append(fixed - truth)
        err_gate.append(gated - truth)
        raw_predictions.append((sel, sp_post, truth))
        has_gbdt = gbdt is not None and len(gbdt) == len(sel)
        if has_gbdt:
            final_sel = 0.3 * gbdt + 0.7 * sel
            final_fixed = 0.3 * gbdt + 0.7 * fixed
            final_gate = 0.3 * gbdt + 0.7 * gated
            err_final_sel.append(final_sel - truth)
            err_final_fixed.append(final_fixed - truth)
            err_final_gate.append(final_gate - truth)
        rows.append(
            {
                "well": wid,
                "n_eval": len(truth),
                "sel_rmse": rmse(sel - truth),
                "sp_rmse": rmse(sp_post - truth),
                "fixed_rmse": rmse(fixed - truth),
                "gate_rmse": rmse(gated - truth),
                "gate_weight": sp.gate_weight,
                "median_dist": sp.median_dist,
                "best_tail_rmse": float(np.nanmin(sp.tail_rmse)),
                "best_surface": SURFACES[int(np.nanargmin(sp.tail_rmse))],
                "known_rmse_min": float(np.nanmin(sp.known_rmse)),
                "known_rmse_egfdu": float(sp.known_rmse[SURFACES.index("EGFDU")]),
                "tail_rmse_egfdu": float(sp.tail_rmse[SURFACES.index("EGFDU")]),
                "has_gbdt": has_gbdt,
            }
        )

    if not rows:
        raise RuntimeError("No wells evaluated")

    def pooled(parts: list[np.ndarray]) -> float:
        return rmse(np.concatenate(parts))

    print("")
    print(f"Evaluated wells: {len(rows)} mode={mode} k={k}")
    print(f"selector       {pooled(err_sel):.4f}")
    print(f"spatial only   {pooled(err_sp):.4f}")
    print(f"fixed w={fixed_weight:.3f} {pooled(err_fixed):.4f}")
    print(f"gated max={max_weight:.3f} {pooled(err_gate):.4f}")
    if err_final_sel:
        print("")
        print(f"Final blend rows with GBDT: {len(err_final_sel)} wells")
        print(f"final selector       {pooled(err_final_sel):.4f}")
        print(f"final fixed w={fixed_weight:.3f} {pooled(err_final_fixed):.4f}")
        print(f"final gated max={max_weight:.3f} {pooled(err_final_gate):.4f}")

    out = pd.DataFrame(rows)
    print("")
    print("Gate summary:")
    print(out[["gate_weight", "best_tail_rmse", "median_dist"]].describe().to_string())
    print("")
    print("Best surfaces:")
    print(out["best_surface"].value_counts().to_string())

    if sweep:
        print("")
        print("Fixed-weight sweep:")
        for w in np.arange(0.0, 0.201, 0.02):
            parts = [
                ((1.0 - w) * sel + w * sp_post) - truth
                for sel, sp_post, truth in raw_predictions
            ]
            print(f"w={w:.2f}  {pooled(parts):.4f}")
        if err_final_sel and gbdt_cache:
            print("")
            print("Final fixed-weight sweep:")
            for w in np.arange(0.0, 0.201, 0.02):
                parts = []
                for row, (sel, sp_post, truth) in zip(rows, raw_predictions):
                    gbdt = gbdt_cache.get(row["well"])
                    if gbdt is None or len(gbdt) != len(sel):
                        continue
                    sel_mix = (1.0 - w) * sel + w * sp_post
                    parts.append((0.3 * gbdt + 0.7 * sel_mix) - truth)
                print(f"w={w:.2f}  {pooled(parts):.4f}")

    if save:
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = paths.output_dir / (
            f"spatial_gate_seed{seed}_n{len(rows)}_{mode}_k{k}_stride{stride}.csv"
        )
        out.to_csv(out_path, index=False)
        print("")
        print(f"Saved per-well diagnostics: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a gated spatial branch without modifying ../rogii.")
    parser.add_argument("--n", type=int, default=80, help="Number of train wells to sample. Omit by passing 0 for all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--mode", choices=["egfdu", "best", "weighted"], default="weighted")
    parser.add_argument("--fixed-weight", type=float, default=0.08)
    parser.add_argument("--max-weight", type=float, default=0.12)
    parser.add_argument("--err-scale", type=float, default=5.0)
    parser.add_argument("--dist-scale", type=float, default=500.0)
    parser.add_argument("--tail-points", type=int, default=200)
    parser.add_argument("--fetch-extra", type=int, default=400)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    evaluate(
        n=args.n or None,
        seed=args.seed,
        stride=args.stride,
        k=args.k,
        mode=args.mode,
        fixed_weight=args.fixed_weight,
        max_weight=args.max_weight,
        err_scale=args.err_scale,
        dist_scale=args.dist_scale,
        tail_points=args.tail_points,
        fetch_extra=args.fetch_extra,
        sweep=args.sweep,
        save=args.save,
    )


if __name__ == "__main__":
    main()
