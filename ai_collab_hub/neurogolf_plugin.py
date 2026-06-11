from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

try:
    from .config import load_config
    from .database import SessionLocal, Project, Topic, Agent, NeuroGolfArtifact, ActivityLog
except ImportError:
    from config import load_config
    from database import SessionLocal, Project, Topic, Agent, NeuroGolfArtifact, ActivityLog


router = APIRouter(prefix="/api/project_plugin", tags=["project-plugin"])

CONFIG = load_config()
WORKSPACE_ROOT = Path(CONFIG["workspace"]["root"]).resolve()
TASK_RE = re.compile(r"^task(\d{3})\.onnx$")
TASK_ID_RE = re.compile(r"task\s*0*(\d{1,3})", re.IGNORECASE)
SOLVED_STATUS = "IS_READY"
DUMMY_SIZE_BYTES = 1024


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def iso(dt) -> str:
    return dt.isoformat() + "Z"


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def normalize_task_id(task_id: str) -> str:
    raw = task_id.strip().lower()
    if raw.startswith("task"):
        raw = raw[4:]
    if not raw.isdigit():
        raise HTTPException(status_code=400, detail=f"非法 task_id: {task_id!r}")
    n = int(raw)
    if not (1 <= n <= 400):
        raise HTTPException(status_code=400, detail="task_id 必须在 task001..task400 范围内。")
    return f"task{n:03d}"


def require_project(db: Session, project: str) -> Project:
    p = db.query(Project).filter(Project.name == project).first()
    if not p:
        raise HTTPException(status_code=404, detail=f"项目 '{project}' 不存在。")
    if p.name != "neurogolf":
        raise HTTPException(status_code=404, detail=f"项目 '{project}' 没有 NeuroGolf 插件。")
    return p


def shared_dir(project: str) -> Path:
    return WORKSPACE_ROOT / project


def working_dir(project: str) -> Path:
    return shared_dir(project) / "data" / "working"


def raw_dir(project: str) -> Path:
    return shared_dir(project) / "data" / "raw"


def manifest_path(project: str) -> Path:
    return working_dir(project) / "solution_manifest.json"


def archive_dir(project: str) -> Path:
    return working_dir(project) / "archive"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_dummy_model(path: Path) -> bool:
    return (not path.exists()) or path.stat().st_size <= DUMMY_SIZE_BYTES


def load_manifest(project: str) -> dict:
    path = manifest_path(project)
    if not path.exists():
        return {"tasks": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 solution_manifest.json 失败: {exc}")
    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="solution_manifest.json 必须是 JSON object。")
    data.setdefault("tasks", {})
    if not isinstance(data["tasks"], dict):
        raise HTTPException(status_code=500, detail="solution_manifest.json 的 tasks 字段必须是 object。")
    return data


def save_manifest(project: str, data: dict) -> None:
    path = manifest_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def deployed_artifacts(db: Session, project_id: int) -> dict[str, NeuroGolfArtifact]:
    rows = (db.query(NeuroGolfArtifact)
            .filter(NeuroGolfArtifact.project_id == project_id,
                    NeuroGolfArtifact.is_deployed == True)  # noqa: E712
            .all())
    return {row.task_id: row for row in rows}


def latest_forum_by_task(db: Session, project_id: int) -> dict[str, dict]:
    names = {a.id: a.name for a in db.query(Agent).all()}
    topics = (db.query(Topic)
              .filter(Topic.project_id == project_id)
              .order_by(Topic.created_at.desc())
              .all())
    out: dict[str, dict] = {}
    for t in topics:
        text = "\n".join([t.title or "", t.content or "", t.conclusion or ""])
        for n in TASK_ID_RE.findall(text):
            tid = f"task{int(n):03d}"
            if tid not in out:
                status = "已完结" if t.closed_at is not None else "待执行" if t.claimed_by_id else "验证提案"
                out[tid] = {
                    "topic_id": t.id,
                    "status": status,
                    "creator": names.get(t.creator_id, "Unknown"),
                    "claimed_by": names.get(t.claimed_by_id) if t.claimed_by_id else None,
                }
    return out


def read_task_index(project: str) -> dict[str, dict]:
    path = working_dir(project) / "task_index.csv"
    if not path.exists():
        return {}
    import csv
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("task_id") or row.get("task") or row.get("id")
            if not raw:
                continue
            try:
                tid = normalize_task_id(Path(raw).stem)
            except HTTPException:
                continue
            rows[tid] = row
    return rows


def task_file(project: str, task_id: str) -> Path:
    return working_dir(project) / f"{task_id}.onnx"


def artifact_status(row: Optional[NeuroGolfArtifact], manifest_entry: Optional[dict], path: Path) -> dict:
    exists = path.exists()
    dummy = is_dummy_model(path)
    if not row:
        manifest_entry = manifest_entry or {}
        manifest_verified = manifest_entry.get("verified_status") or "UNKNOWN"
        manifest_sha = manifest_entry.get("model_sha256")
        manifest_score = manifest_entry.get("deployed_score", manifest_entry.get("best_score"))
        manifest_topic = manifest_entry.get("source_topic")
        manifest_deployed = manifest_verified == SOLVED_STATUS and exists and not dummy
        return {
            "verified_status": manifest_verified,
            "score": manifest_score,
            "sha256": manifest_sha or (sha256_file(path) if exists else None),
            "bytes": path.stat().st_size if exists else 0,
            "is_deployed": manifest_deployed,
            "is_dummy": dummy,
            "source_topic": manifest_topic,
            "created_by": manifest_entry.get("created_by"),
            "artifact_age": None,
        }
    age = None
    if row.updated_at:
        age = (datetime.utcnow() - row.updated_at).total_seconds()
    return {
        "verified_status": row.verified_status,
        "score": row.score,
        "sha256": row.sha256,
        "bytes": row.bytes,
        "is_deployed": row.is_deployed,
        "is_dummy": row.is_dummy,
        "source_topic": row.forum_topic_id,
        "created_by": row.created_by,
        "artifact_age": age,
    }


def verify_model(project: str, task_id: str, model_path: Path) -> str:
    if os.environ.get("NEUROGOLF_VERIFY_CMD"):
        cmd = os.environ["NEUROGOLF_VERIFY_CMD"].format(
            model=str(model_path),
            task=task_id,
            task_json=str(raw_dir(project) / f"{task_id}.json"),
            raw_dir=str(raw_dir(project)),
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0 and ("IS READY" in output or "READY" in output):
            return SOLVED_STATUS
        raise HTTPException(status_code=422, detail=f"官方验证失败:\n{output[-4000:]}")

    utils = raw_dir(project) / "neurogolf_utils" / "neurogolf_utils.py"
    task_json = raw_dir(project) / f"{task_id}.json"
    if not utils.exists() or not task_json.exists():
        raise HTTPException(
            status_code=503,
            detail="中心机缺少 NeuroGolf raw data 或 neurogolf_utils.py，无法执行官方验证。"
        )

    script = r"""
import importlib.util, inspect, pathlib, sys
utils_path, model_path, task_json, task_id = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("neurogolf_utils", utils_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
func = None
for name in ("verify_network", "verify_model", "benchmark_network", "run_benchmark"):
    if hasattr(mod, name):
        func = getattr(mod, name)
        break
if func is None:
    raise SystemExit("No known verify function in neurogolf_utils.py")
args_list = [
    (model_path, task_json),
    (task_json, model_path),
    (pathlib.Path(model_path), pathlib.Path(task_json)),
    (pathlib.Path(task_json), pathlib.Path(model_path)),
    (model_path, int(task_id[-3:])),
    (int(task_id[-3:]), model_path),
    (model_path,),
]
last = None
for args in args_list:
    try:
        result = func(*args)
        print(result)
        raise SystemExit(0)
    except TypeError as exc:
        last = exc
        continue
raise SystemExit(f"Could not call verifier: {last}")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script, str(utils), str(model_path), str(task_json), task_id],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode == 0 and ("IS READY" in output or "pass" in output.lower()):
        return SOLVED_STATUS
    raise HTTPException(status_code=422, detail=f"官方验证失败:\n{output[-4000:]}")


def archive_current(project: str, task_id: str, current: Path, deployed: Optional[NeuroGolfArtifact]) -> Optional[Path]:
    if not current.exists():
        return None
    archive_dir(project).mkdir(parents=True, exist_ok=True)
    score = "unknown" if not deployed or deployed.score is None else f"{deployed.score:.3f}"
    topic = "notopic" if not deployed or deployed.forum_topic_id is None else f"t{deployed.forum_topic_id}"
    digest = sha256_file(current)[:12]
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = archive_dir(project) / f"{task_id}_{score}pts_{topic}_{digest}_{stamp}.onnx"
    shutil.move(str(current), str(target))
    return target


def rebuild_submission_zip(project: str) -> dict:
    wd = working_dir(project)
    zip_path = wd / "submission.zip"
    missing = [f"task{i:03d}.onnx" for i in range(1, 401) if not (wd / f"task{i:03d}.onnx").exists()]
    if missing:
        raise HTTPException(status_code=409, detail=f"缺少 {len(missing)} 个 ONNX，不能重建 submission.zip: {missing[:8]}")

    try:
        import onnx  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"缺少 onnx 依赖，无法做 zip 内加载校验: {exc}")

    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(1, 401):
            path = wd / f"task{i:03d}.onnx"
            zf.write(path, arcname=path.name)

    bad: list[str] = []
    with zipfile.ZipFile(tmp, "r") as zf:
        for name in zf.namelist():
            try:
                onnx.load_model_from_string(zf.read(name))
            except Exception as exc:
                bad.append(f"{name}: {exc}")
    if bad:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"zip 内模型加载失败: {bad[:5]}")
    tmp.replace(zip_path)
    return {"zip": str(zip_path), "checked": 400, "size": zip_path.stat().st_size}


def ensure_submission_inputs(project: str, candidate_task: str) -> None:
    wd = working_dir(project)
    missing = [
        f"task{i:03d}.onnx"
        for i in range(1, 401)
        if f"task{i:03d}" != candidate_task and not (wd / f"task{i:03d}.onnx").exists()
    ]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"缺少 {len(missing)} 个既有 ONNX，部署后无法重建 submission.zip: {missing[:8]}"
        )


@router.get("/{project}/status")
def neurogolf_status(project: str, db: Session = Depends(get_db)):
    p = require_project(db, project)
    artifacts = deployed_artifacts(db, p.id)
    forum = latest_forum_by_task(db, p.id)
    task_index = read_task_index(project)
    manifest_tasks = load_manifest(project).get("tasks", {})
    tasks = []
    counts = {"solved": 0, "attacking": 0, "dummy": 0, "archived": 0}
    for i in range(1, 401):
        tid = f"task{i:03d}"
        path = task_file(project, tid)
        row = artifacts.get(tid)
        art = artifact_status(row, manifest_tasks.get(tid), path)
        f = forum.get(tid)
        idx = task_index.get(tid, {})
        solved = art["verified_status"] == SOLVED_STATUS and art["is_deployed"] and not art["is_dummy"]
        attacking = bool(f and f["status"] in ("待执行", "验证提案") and not solved)
        archived = bool(f and f["status"] == "已完结" and not solved)
        dummy = not solved and not attacking and not archived
        if solved:
            card_status = "solved"; counts["solved"] += 1
        elif attacking:
            card_status = "attacking"; counts["attacking"] += 1
        elif archived:
            card_status = "archived"; counts["archived"] += 1
        else:
            card_status = "dummy"; counts["dummy"] += 1
        tasks.append({
            "id": tid,
            "rule_family": idx.get("rule_family") or idx.get("shape_category") or "UNKNOWN",
            "shape": idx.get("shape") or idx.get("notes") or "VARIABLE",
            "onnx_exists": path.exists(),
            "onnx_size": path.stat().st_size if path.exists() else 0,
            "is_dummy": art["is_dummy"],
            "status": card_status,
            "solved": solved,
            "verified_status": art["verified_status"],
            "best_score": art["score"],
            "deployed_score": art["score"] if art["is_deployed"] else None,
            "sha256": art["sha256"],
            "sha256_short": art["sha256"][:8] if art["sha256"] else None,
            "source_topic": art["source_topic"],
            "created_by": art["created_by"],
            "artifact_age": art["artifact_age"],
            "forum": f,
        })
    return {"tasks": tasks, "counts": counts, "manifest": str(manifest_path(project))}


@router.get("/{project}/artifact/{filename}")
def download_artifact(project: str, filename: str, db: Session = Depends(get_db)):
    require_project(db, project)
    if filename != "submission.zip" and not TASK_RE.match(filename):
        raise HTTPException(status_code=400, detail="只允许下载 submission.zip 或 taskXXX.onnx。")
    path = (working_dir(project) / filename).resolve()
    base = working_dir(project).resolve()
    if base not in path.parents and path != base:
        raise HTTPException(status_code=400, detail="非法 artifact 路径。")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"artifact 不存在: {filename}")
    return FileResponse(path, filename=filename)


@router.get("/{project}/submission")
def download_submission(project: str, db: Session = Depends(get_db)):
    require_project(db, project)
    return download_artifact(project, "submission.zip", db)


@router.post("/{project}/deploy")
async def deploy_artifact(
    project: str,
    file: UploadFile = File(...),
    task_id: str = Form(...),
    score: float = Form(...),
    forum_topic_id: Optional[int] = Form(None),
    agent_name: Optional[str] = Form(None),
    allow_regression: bool = Form(False),
    db: Session = Depends(get_db),
):
    p = require_project(db, project)
    tid = normalize_task_id(task_id)
    expected_name = f"{tid}.onnx"
    if file.filename and Path(file.filename).name != expected_name:
        raise HTTPException(status_code=400, detail=f"上传文件名必须是 {expected_name}。")
    wd = working_dir(project)
    wd.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        candidate = Path(tmpdir) / expected_name
        with candidate.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        if is_dummy_model(candidate):
            raise HTTPException(status_code=422, detail="拒绝部署 dummy/空模型。")
        verified_status = verify_model(project, tid, candidate)
        digest = sha256_file(candidate)
        size = candidate.stat().st_size

        current = (db.query(NeuroGolfArtifact)
                   .filter(NeuroGolfArtifact.project_id == p.id,
                           NeuroGolfArtifact.task_id == tid,
                           NeuroGolfArtifact.is_deployed == True)  # noqa: E712
                   .first())
        historical_scores = [
            r.score for r in db.query(NeuroGolfArtifact)
            .filter(NeuroGolfArtifact.project_id == p.id,
                    NeuroGolfArtifact.task_id == tid,
                    NeuroGolfArtifact.score != None)  # noqa: E711
            .all()
        ]
        previous_best = max(historical_scores) if historical_scores else None
        if previous_best is not None and score < previous_best and not allow_regression:
            raise HTTPException(status_code=409, detail=f"拒绝低分覆盖: {score:.3f} < 历史部署 {previous_best:.3f}")
        ensure_submission_inputs(project, tid)

        archive_path = archive_current(project, tid, task_file(project, tid), current)
        dest = task_file(project, tid)
        shutil.copy2(candidate, dest)
        if current:
            current.is_deployed = False
            current.updated_at = utcnow()

        row = NeuroGolfArtifact(
            project_id=p.id,
            task_id=tid,
            score=score,
            verified_status=verified_status,
            sha256=digest,
            bytes=size,
            forum_topic_id=forum_topic_id,
            created_by=agent_name,
            artifact_path=str(dest),
            is_deployed=True,
            is_dummy=False,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        db.add(row)

        manifest = load_manifest(project)
        manifest_best = max(previous_best, score) if previous_best is not None else score
        manifest["tasks"][tid] = {
            "best_score": manifest_best,
            "deployed_score": score,
            "verified_status": verified_status,
            "source_topic": forum_topic_id,
            "created_by": agent_name,
            "model_sha256": digest,
            "model_path": str(dest),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        save_manifest(project, manifest)
        try:
            zip_info = rebuild_submission_zip(project)
        except HTTPException:
            dest.unlink(missing_ok=True)
            if archive_path:
                shutil.move(str(archive_path), str(dest))
            raise
        db.add(ActivityLog(project_id=p.id, agent_id=None, action_type="artifact_deploy",
                           topic_id=forum_topic_id,
                           description=f"{tid} 部署成功: score={score}, sha256={digest[:8]}"))
        db.commit()
        db.refresh(row)
        return {
            "status": "success",
            "artifact_id": row.id,
            "task_id": tid,
            "score": score,
            "verified_status": verified_status,
            "sha256": digest,
            "bytes": size,
            "archived": str(archive_path) if archive_path else None,
            "submission": zip_info,
        }


@router.post("/{project}/submit")
def submit_kaggle(project: str, message: str = Form("NeuroGolf Hub submission"),
                  db: Session = Depends(get_db)):
    require_project(db, project)
    if os.environ.get("NEUROGOLF_ENABLE_SERVER_SUBMIT") != "1":
        raise HTTPException(status_code=403, detail="中心机 Kaggle 提交默认关闭。设置 NEUROGOLF_ENABLE_SERVER_SUBMIT=1 才可用。")
    zip_path = working_dir(project) / "submission.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="submission.zip 不存在。")
    proc = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", "neurogolf-2026", "-f", str(zip_path), "-m", message],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=(proc.stdout + "\n" + proc.stderr)[-4000:])
    return {"status": "success", "output": proc.stdout}
