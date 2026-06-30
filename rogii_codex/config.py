from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    rogii_dir: Path
    data_dir: Path
    train_dir: Path
    test_dir: Path
    selector_cache: Path
    train_side_cache: Path
    output_dir: Path


def default_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    rogii_dir = repo_root / "rogii"
    data_dir = rogii_dir / "data" / "extracted"
    return Paths(
        repo_root=repo_root,
        rogii_dir=rogii_dir,
        data_dir=data_dir,
        train_dir=data_dir / "train",
        test_dir=data_dir / "test",
        selector_cache=rogii_dir / "selcache.pkl",
        train_side_cache=rogii_dir / "train_side.pkl",
        output_dir=repo_root / "rogii_codex" / "outputs",
    )
