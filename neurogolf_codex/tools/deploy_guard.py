from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DeployError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeployResult:
    task: str
    deployed_path: Path
    score: float
    sha256: str
    archived_paths: tuple[Path, ...]
    dry_run: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_task(task: str) -> str:
    raw = task.strip().lower()
    if raw.startswith("task"):
        raw = raw[4:]
    if not raw.isdigit():
        raise DeployError(f"invalid task id: {task!r}")
    return f"task{int(raw):03d}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"tasks": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise DeployError(f"manifest must be a JSON object: {path}")
    data.setdefault("tasks", {})
    if not isinstance(data["tasks"], dict):
        raise DeployError("manifest field 'tasks' must be an object")
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def candidate_sidecars(model_path: Path) -> list[Path]:
    candidates = [
        model_path.with_name(model_path.name + ".data"),
        model_path.with_suffix(model_path.suffix + ".data"),
    ]
    return sorted({p for p in candidates if p.exists()})


def _score_value(entry: dict[str, Any], key: str) -> float | None:
    value = entry.get(key)
    if value is None:
        return None
    return float(value)


def current_best_score(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    scores = [
        _score_value(entry, "best_score"),
        _score_value(entry, "deployed_score"),
    ]
    present = [s for s in scores if s is not None]
    return max(present) if present else None


def archive_existing(dest: Path, archive_dir: Path, task: str, old_entry: dict[str, Any] | None) -> tuple[Path, ...]:
    if not dest.exists():
        return ()
    archive_dir.mkdir(parents=True, exist_ok=True)
    old_score = old_entry.get("deployed_score") if old_entry else None
    old_sha = sha256_file(dest)[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    score_part = "unknown" if old_score is None else f"{float(old_score):.3f}"
    base = f"{task}_{score_part}pts_{old_sha}_{stamp}"
    archived: list[Path] = []

    target = archive_dir / f"{base}.onnx"
    shutil.move(str(dest), str(target))
    archived.append(target)

    for sidecar in candidate_sidecars(dest):
        side_target = archive_dir / f"{base}{sidecar.suffix}"
        shutil.move(str(sidecar), str(side_target))
        archived.append(side_target)

    return tuple(archived)


def deploy_model(
    *,
    task: str,
    model_path: Path,
    score: float,
    source_topic: int | None,
    working_dir: Path,
    manifest_path: Path,
    archive_dir: Path,
    allow_regression: bool = False,
    allow_external_data: bool = False,
    dry_run: bool = False,
) -> DeployResult:
    task_id = normalize_task(task)
    model_path = model_path.resolve()
    if not model_path.exists():
        raise DeployError(f"model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise DeployError(f"model must be an .onnx file: {model_path}")

    sidecars = candidate_sidecars(model_path)
    if sidecars and not allow_external_data:
        joined = ", ".join(str(p) for p in sidecars)
        raise DeployError(
            "candidate uses external data sidecar(s); embed weights before deployment: "
            f"{joined}"
        )

    manifest = load_manifest(manifest_path)
    task_entry = manifest["tasks"].get(task_id)
    previous_best = current_best_score(task_entry)
    if previous_best is not None and score < previous_best and not allow_regression:
        raise DeployError(
            f"refusing regression for {task_id}: new score {score:.3f} < "
            f"historical best {previous_best:.3f}"
        )

    digest = sha256_file(model_path)
    dest = working_dir / f"{task_id}.onnx"

    if dry_run:
        return DeployResult(task_id, dest, score, digest, (), dry_run=True)

    archived = archive_existing(dest, archive_dir, task_id, task_entry)
    working_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, dest)

    now = datetime.now(timezone.utc).isoformat()
    new_best = score if previous_best is None else max(previous_best, score)
    manifest["tasks"][task_id] = {
        "best_score": new_best,
        "deployed_score": score,
        "source_topic": source_topic,
        "model_sha256": digest,
        "model_path": str(dest),
        "updated_at": now,
    }
    save_manifest(manifest_path, manifest)

    return DeployResult(task_id, dest, score, digest, archived)


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    working = root / "neurogolf" / "data" / "working"
    parser = argparse.ArgumentParser(
        description="Deploy a NeuroGolf ONNX only if it does not regress the recorded best score."
    )
    parser.add_argument("--task", required=True, help="Task id, e.g. 15 or task015.")
    parser.add_argument("--model", required=True, type=Path, help="Candidate .onnx path.")
    parser.add_argument("--score", required=True, type=float, help="Official local score.")
    parser.add_argument("--topic", type=int, default=None, help="Forum topic id that justifies the model.")
    parser.add_argument("--working-dir", type=Path, default=working)
    parser.add_argument("--manifest", type=Path, default=working / "solution_manifest.json")
    parser.add_argument("--archive-dir", type=Path, default=working / "archive")
    parser.add_argument("--allow-regression", action="store_true")
    parser.add_argument("--allow-external-data", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = deploy_model(
            task=args.task,
            model_path=args.model,
            score=args.score,
            source_topic=args.topic,
            working_dir=args.working_dir,
            manifest_path=args.manifest,
            archive_dir=args.archive_dir,
            allow_regression=args.allow_regression,
            allow_external_data=args.allow_external_data,
            dry_run=args.dry_run,
        )
    except DeployError as exc:
        print(f"ERROR: {exc}")
        return 2

    payload = {
        "task": result.task,
        "deployed_path": str(result.deployed_path),
        "score": result.score,
        "sha256": result.sha256,
        "archived_paths": [str(p) for p in result.archived_paths],
        "dry_run": result.dry_run,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
