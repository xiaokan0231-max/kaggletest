from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import default_paths


FULL_OBS_COLS = ("MD", "X", "Y", "Z", "GR")
PREFIX_COLS = ("TVT_input",)


@dataclass
class MatchMetrics:
    train_well: str
    n_rows: int
    ps: int
    full_rmse: float
    full_max: float
    md_max: float
    xy_max: float
    z_max: float
    gr_max: float
    tvt_input_max: float
    nan_mismatch: int
    score: float
    triggered: bool


def well_id_from_path(path: Path) -> str:
    return path.name.split("__")[0]


def ps_count(df: pd.DataFrame) -> int:
    return int(df["TVT_input"].notna().sum())


def train_files(train_dir: Path) -> list[Path]:
    return sorted(train_dir.glob("*__horizontal_well.csv"))


def test_files(test_dir: Path) -> list[Path]:
    return sorted(test_dir.glob("*__horizontal_well.csv"))


def load_well(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _nan_aware_diff(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    mismatch = int(np.count_nonzero(a_nan != b_nan))
    both = ~(a_nan | b_nan)
    diff = np.zeros_like(a, dtype=float)
    diff[both] = a[both] - b[both]
    diff[a_nan & b_nan] = 0.0
    diff[a_nan != b_nan] = np.inf
    return diff, mismatch


def _max_abs(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    if np.isinf(diff).any():
        return float("inf")
    return float(np.max(np.abs(diff)))


def _rmse(diff: np.ndarray) -> float:
    if diff.size == 0:
        return 0.0
    if np.isinf(diff).any():
        return float("inf")
    return float(np.sqrt(np.mean(diff * diff)))


def observable_digest(df: pd.DataFrame, decimals: int = 5) -> str:
    """Hash only submission-time observable columns.

    This is an exact duplicate detector after rounding. It intentionally ignores
    train-only columns such as TVT and formation surfaces.
    """
    ps = ps_count(df)
    parts: list[bytes] = [str(len(df)).encode(), b"|", str(ps).encode(), b"|"]
    for col in FULL_OBS_COLS:
        arr = df[col].to_numpy(dtype=float)
        rounded = np.round(arr, decimals=decimals)
        rounded = np.nan_to_num(rounded, nan=-999999999.0)
        parts.append(rounded.astype("<f8", copy=False).tobytes())
        parts.append(b"|")
    for col in PREFIX_COLS:
        arr = df[col].to_numpy(dtype=float)[:ps]
        rounded = np.round(arr, decimals=decimals)
        rounded = np.nan_to_num(rounded, nan=-999999999.0)
        parts.append(rounded.astype("<f8", copy=False).tobytes())
        parts.append(b"|")
    return hashlib.sha256(b"".join(parts)).hexdigest()


def compare_wells(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    train_well: str,
    thresholds: dict[str, float],
) -> MatchMetrics | None:
    if len(test_df) != len(train_df):
        return None
    ps = ps_count(test_df)
    if ps != ps_count(train_df):
        return None

    nan_mismatch = 0
    full_diffs = []
    per_col_max = {}
    for col in FULL_OBS_COLS:
        diff, mism = _nan_aware_diff(test_df[col].to_numpy(), train_df[col].to_numpy())
        nan_mismatch += mism
        full_diffs.append(diff)
        per_col_max[col] = _max_abs(diff)

    tvt_diff, mism = _nan_aware_diff(
        test_df["TVT_input"].to_numpy(dtype=float)[:ps],
        train_df["TVT_input"].to_numpy(dtype=float)[:ps],
    )
    nan_mismatch += mism

    full = np.concatenate(full_diffs) if full_diffs else np.array([], dtype=float)
    full_rmse = _rmse(full)
    full_max = _max_abs(full)
    tvt_input_max = _max_abs(tvt_diff)

    xy_max = max(per_col_max["X"], per_col_max["Y"])
    score_parts = [
        per_col_max["MD"] / thresholds["md_max"],
        xy_max / thresholds["xy_max"],
        per_col_max["Z"] / thresholds["z_max"],
        per_col_max["GR"] / thresholds["gr_max"],
        tvt_input_max / thresholds["tvt_input_max"],
    ]
    score = float(max(score_parts))
    triggered = (
        nan_mismatch == 0
        and per_col_max["MD"] <= thresholds["md_max"]
        and xy_max <= thresholds["xy_max"]
        and per_col_max["Z"] <= thresholds["z_max"]
        and per_col_max["GR"] <= thresholds["gr_max"]
        and tvt_input_max <= thresholds["tvt_input_max"]
    )

    return MatchMetrics(
        train_well=train_well,
        n_rows=len(test_df),
        ps=ps,
        full_rmse=full_rmse,
        full_max=full_max,
        md_max=per_col_max["MD"],
        xy_max=xy_max,
        z_max=per_col_max["Z"],
        gr_max=per_col_max["GR"],
        tvt_input_max=tvt_input_max,
        nan_mismatch=nan_mismatch,
        score=score,
        triggered=triggered,
    )


def default_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "md_max": args.md_max,
        "xy_max": args.xy_max,
        "z_max": args.z_max,
        "gr_max": args.gr_max,
        "tvt_input_max": args.tvt_input_max,
    }


def build_train_index(train_dir: Path, decimals: int) -> tuple[dict[str, Path], dict[str, list[str]], dict[tuple[int, int], list[str]]]:
    paths = {well_id_from_path(path): path for path in train_files(train_dir)}
    digest_index: dict[str, list[str]] = {}
    block_index: dict[tuple[int, int], list[str]] = {}

    for wid, path in paths.items():
        df = load_well(path)
        digest_index.setdefault(observable_digest(df, decimals=decimals), []).append(wid)
        block_index.setdefault((len(df), ps_count(df)), []).append(wid)

    return paths, digest_index, block_index


def top_matches_for_test(
    test_well: str,
    test_df: pd.DataFrame,
    train_paths: dict[str, Path],
    block_index: dict[tuple[int, int], list[str]],
    thresholds: dict[str, float],
    exclude_same_id: bool,
    limit: int,
) -> list[MatchMetrics]:
    block = (len(test_df), ps_count(test_df))
    candidates = block_index.get(block, [])
    out: list[MatchMetrics] = []

    for train_well in candidates:
        if exclude_same_id and train_well == test_well:
            continue
        train_df = load_well(train_paths[train_well])
        metrics = compare_wells(test_df, train_df, train_well, thresholds)
        if metrics is not None:
            out.append(metrics)

    out.sort(key=lambda row: (not row.triggered, row.score, row.full_rmse, row.train_well))
    return out[:limit]


def local_copy_rmse(test_well: str, match_well: str) -> float | None:
    paths = default_paths()
    train_path = paths.train_dir / f"{match_well}__horizontal_well.csv"
    truth_path = paths.train_dir / f"{test_well}__horizontal_well.csv"
    test_path = paths.test_dir / f"{test_well}__horizontal_well.csv"
    if not (train_path.exists() and truth_path.exists() and test_path.exists()):
        return None
    train_df = load_well(train_path)
    truth_df = load_well(truth_path)
    test_df = load_well(test_path)
    ps = ps_count(test_df)
    if len(train_df) != len(test_df) or len(truth_df) != len(test_df):
        return None
    pred = train_df["TVT"].to_numpy(dtype=float)[ps:]
    truth = truth_df["TVT"].to_numpy(dtype=float)[ps:]
    return _rmse(pred - truth)


def audit_test(args: argparse.Namespace) -> list[dict]:
    paths = default_paths()
    thresholds = default_thresholds(args)
    train_paths, digest_index, block_index = build_train_index(paths.train_dir, args.digest_decimals)

    reports = []
    for path in test_files(paths.test_dir):
        test_well = well_id_from_path(path)
        test_df = load_well(path)
        digest_hits = digest_index.get(observable_digest(test_df, decimals=args.digest_decimals), [])
        if args.exclude_same_id:
            digest_hits = [wid for wid in digest_hits if wid != test_well]

        matches = top_matches_for_test(
            test_well=test_well,
            test_df=test_df,
            train_paths=train_paths,
            block_index=block_index,
            thresholds=thresholds,
            exclude_same_id=args.exclude_same_id,
            limit=args.top,
        )
        triggered = [m for m in matches if m.triggered]
        chosen = triggered[0] if triggered else None
        copy_rmse = local_copy_rmse(test_well, chosen.train_well) if chosen else None
        reports.append(
            {
                "test_well": test_well,
                "n_rows": len(test_df),
                "ps": ps_count(test_df),
                "digest_hits": digest_hits,
                "triggered": bool(chosen),
                "chosen_train_well": chosen.train_well if chosen else None,
                "copy_rmse_if_visible": copy_rmse,
                "top_matches": [asdict(m) for m in matches],
            }
        )
    return reports


def audit_train_duplicates(args: argparse.Namespace) -> dict:
    paths = default_paths()
    train_paths, digest_index, _ = build_train_index(paths.train_dir, args.digest_decimals)
    duplicate_groups = {dig: wells for dig, wells in digest_index.items() if len(wells) > 1}
    return {
        "train_wells": len(train_paths),
        "duplicate_groups": duplicate_groups,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_well_count": sum(len(wells) for wells in duplicate_groups.values()),
    }


def print_test_report(reports: list[dict]) -> None:
    for report in reports:
        print("")
        print(
            f"test={report['test_well']} rows={report['n_rows']} ps={report['ps']} "
            f"triggered={report['triggered']} chosen={report['chosen_train_well']}"
        )
        print(f"digest_hits={report['digest_hits']}")
        if report["copy_rmse_if_visible"] is not None:
            print(f"copy_rmse_if_visible={report['copy_rmse_if_visible']:.6f}")
        for i, match in enumerate(report["top_matches"], start=1):
            print(
                f"  #{i} train={match['train_well']} trigger={match['triggered']} "
                f"score={match['score']:.6g} full_rmse={match['full_rmse']:.6g} "
                f"md_max={match['md_max']:.6g} xy_max={match['xy_max']:.6g} "
                f"z_max={match['z_max']:.6g} gr_max={match['gr_max']:.6g} "
                f"tvt_input_max={match['tvt_input_max']:.6g} nan_mismatch={match['nan_mismatch']}"
            )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-op-safe train/test fingerprint matcher. Reads ../rogii only."
    )
    parser.add_argument("--mode", choices=["test", "train-dupes", "all"], default="all")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--exclude-same-id", action="store_true")
    parser.add_argument("--digest-decimals", type=int, default=5)
    parser.add_argument("--md-max", type=float, default=1e-3)
    parser.add_argument("--xy-max", type=float, default=1e-3)
    parser.add_argument("--z-max", type=float, default=1e-3)
    parser.add_argument("--gr-max", type=float, default=1e-3)
    parser.add_argument("--tvt-input-max", type=float, default=1e-3)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    paths = default_paths()
    result = {}
    if args.mode in ("test", "all"):
        reports = audit_test(args)
        result["test"] = reports
        print_test_report(reports)
    if args.mode in ("train-dupes", "all"):
        dupes = audit_train_duplicates(args)
        result["train_duplicates"] = dupes
        print("")
        print(
            f"train duplicate groups={dupes['duplicate_group_count']} "
            f"duplicate wells={dupes['duplicate_well_count']} / {dupes['train_wells']}"
        )
        if dupes["duplicate_groups"]:
            for wells in dupes["duplicate_groups"].values():
                print("  " + ", ".join(wells))

    if args.save:
        suffix = "exclude_same" if args.exclude_same_id else "include_same"
        out_path = paths.output_dir / f"fingerprint_audit_{args.mode}_{suffix}.json"
        write_json(out_path, result)
        print("")
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

