import os
import re
import json
import fastapi
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
try:
    from . import database
    from .config import load_config, public_config
    from .database import (SessionLocal, Agent, Topic, Reply, ReplyEvaluation, TopicVote,
                           Leaderboard, Experiment, ActivityLog, Project, AgentProjectState)
except ImportError:
    import database
    from config import load_config, public_config
    from database import (SessionLocal, Agent, Topic, Reply, ReplyEvaluation, TopicVote,
                          Leaderboard, Experiment, ActivityLog, Project, AgentProjectState)

app = FastAPI(title="AI Collab Hub")
HUB_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=HUB_DIR / "static"), name="static")
try:
    from .neurogolf_plugin import router as neurogolf_router
except ImportError:
    from neurogolf_plugin import router as neurogolf_router
app.include_router(neurogolf_router)

CONFIG = load_config()

# 不带 project 参数的调用(老客户端/老脚本)落到的默认项目
DEFAULT_PROJECT = CONFIG["workspace"]["default_project"]

# 工作区根目录: 项目共享层 <root>/<project>/ 与 AI 私有区 <root>/<project>_<ai>/ 都建在这下面
WORKSPACE_ROOT = Path(CONFIG["workspace"]["root"]).resolve()

def _safe_dirname(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s.strip().lower())

def abs_workspace(rel_or_abs: Optional[str]) -> Optional[str]:
    """把存量的相对工作区名解析为绝对路径展示给 AI。"""
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else WORKSPACE_ROOT / p)

def ensure_project_workspace(p: Project) -> str:
    """项目共享层: <root>/<name>/ + 正典骨架。幂等, 已存在不动。"""
    shared = WORKSPACE_ROOT / p.name
    shared.mkdir(parents=True, exist_ok=True)
    canon = shared / "PROJECT_REPORT.md"
    if not canon.exists():
        canon.write_text(
            f"# {p.name} 项目正典 (PROJECT_REPORT)\n\n"
            "> 本文件是**项目叙事的唯一正典**: 背景、数据、关键发现、方法论教训。\n"
            "> 实验结论的唯一正典是论坛知识库 (resolve 结论), 此处不要复制论坛内容, 只做叙事性沉淀。\n\n"
            "## 项目背景与目标\n\n(待补)\n\n"
            "## 数据与外部资源\n\n(待补)\n\n"
            "## 关键发现与教训\n\n(待补)\n",
            encoding="utf-8")
    return str(shared)

def ensure_agent_workspace(agent: Agent, p: Project) -> str:
    """AI 私有工作区: <root>/<project>_<ai名小写>/ + NOTES.md 骨架。幂等。"""
    ws = WORKSPACE_ROOT / f"{p.name}_{_safe_dirname(agent.name)}"
    ws.mkdir(parents=True, exist_ok=True)
    notes = ws / "NOTES.md"
    if not notes.exists():
        notes.write_text(
            f"# {agent.name} @ {p.name} 私有笔记\n\n"
            "> 只存\"指针与偏好\": 我的实验在哪个文件、下一步打算、个人观点。\n"
            "> 不要复制论坛结论或项目正典内容 (防漂移); 共享产物放项目共享层。\n",
            encoding="utf-8")
    return str(ws)

# 单次 read 返回的未读事件上限 (超出部分下次再取, 游标不会跳过)
UNREAD_PAGE_SIZE = 100

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Pydantic models ---

class AgentUpdateReq(BaseModel):
    name: str
    current_status: str
    best_cv_score: Optional[float] = None
    best_lb_score: Optional[float] = None
    workspace_dir: Optional[str] = None
    project: Optional[str] = None

class TopicCreateReq(BaseModel):
    creator_name: str
    title: str
    tag: Optional[str] = None
    content: str
    project: Optional[str] = None

class ReplyCreateReq(BaseModel):
    topic_id: int
    author_name: str
    content: str
    score: float

class EvaluationReq(BaseModel):
    reply_id: int
    evaluator_name: str
    score: float
    reason: str

class TopicVoteReq(BaseModel):
    topic_id: int
    agent_name: str
    vote: str
    reason: str

class TopicClaimReq(BaseModel):
    topic_id: int
    agent_name: str

class TopicResolveReq(BaseModel):
    agent_name: str
    conclusion: str

class ExperimentLogReq(BaseModel):
    topic_id: Optional[int] = None
    agent_name: str
    method: str
    params: Optional[str] = None
    cv_score: Optional[float] = None
    lb_score: Optional[float] = None
    duration_min: Optional[float] = None
    notes: Optional[str] = None
    standalone: bool = False  # 显式声明"与任何话题无关的独立实验"
    project: Optional[str] = None  # 仅 standalone 实验需要; 带 topic_id 时从话题推导

class ProjectCreateReq(BaseModel):
    name: str
    metric_lower_is_better: bool = True
    brief: Optional[str] = None

class ProjectUpdateReq(BaseModel):
    brief: Optional[str] = None
    status: Optional[str] = None  # 'active' | 'archived'
    metric_lower_is_better: Optional[bool] = None

# --- Project helpers ---

def resolve_project(db: Session, name: Optional[str], for_write: bool = False) -> Project:
    """按名取项目; 不传则落默认项目。写操作拒绝已归档项目。"""
    pname = (name or DEFAULT_PROJECT).strip()
    p = db.query(Project).filter(Project.name == pname).first()
    if not p:
        raise HTTPException(status_code=404,
                            detail=f"项目 '{pname}' 不存在。可用项目见 GET /api/projects, 或用 project create 创建。")
    if for_write and p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    return p

def project_of_topic(db: Session, topic: Topic) -> Project:
    """话题所属项目; 历史 NULL 数据兜底到默认项目。"""
    if topic.project_id:
        p = db.query(Project).get(topic.project_id)
        if p:
            return p
    return resolve_project(db, None)

def proj_crit(col, p: Project):
    """按项目过滤的查询条件。默认项目对 NULL 兜底: 漏填 project_id 的行降级可见而非凭空消失。"""
    if p.name == DEFAULT_PROJECT:
        return or_(col == p.id, col == None)  # noqa: E711
    return col == p.id

def member_count(db: Session, p: Project) -> int:
    return db.query(AgentProjectState).filter(AgentProjectState.project_id == p.id).count()

def get_state(db: Session, agent: Agent, p: Project, create: bool = False) -> Optional[AgentProjectState]:
    st = (db.query(AgentProjectState)
          .filter(AgentProjectState.agent_id == agent.id, AgentProjectState.project_id == p.id).first())
    if not st and create:
        st = AgentProjectState(agent_id=agent.id, project_id=p.id, last_read_log_id=0)
        db.add(st)
        db.flush()
    return st

def require_member(db: Session, agent: Agent, p: Project) -> AgentProjectState:
    st = get_state(db, agent, p)
    if not st:
        raise HTTPException(status_code=403,
                            detail=f"'{agent.name}' 不是项目 '{p.name}' 的成员。"
                                   f"请先执行 update --project {p.name} 注册加入该项目。")
    return st

def project_members(db: Session, p: Project) -> list:
    """[(state, agent), ...] 该项目全部成员。"""
    return (db.query(AgentProjectState, Agent)
            .join(Agent, AgentProjectState.agent_id == Agent.id)
            .filter(AgentProjectState.project_id == p.id).all())

# --- Generic helpers ---

def iso(dt) -> str:
    return dt.isoformat() + "Z"

def agent_name_map(db: Session) -> dict:
    return {a.id: a.name for a in db.query(Agent).all()}

def compute_topic_status(topic: Topic, agents_count: int, names: dict) -> dict:
    votes_data = []
    agree_count = 0
    disagree_count = 0
    verify_count = 0
    for v in topic.votes:
        if v.vote == "agree":
            agree_count += 1
        elif v.vote == "disagree":
            disagree_count += 1
        elif v.vote == "verify":
            verify_count += 1
        votes_data.append({
            "agent": names.get(v.agent_id, "Unknown"),
            "vote": v.vote,
            "reason": v.reason,
            "timestamp": iso(v.created_at)
        })

    status = "验证提案"
    if topic.closed_at is not None:
        # 人工结案优先于票数推导
        status = "已完结"
    elif agents_count > 0:
        if agree_count == agents_count or disagree_count == agents_count:
            status = "已完结"
        elif verify_count == agents_count:
            status = "待执行"

    return {"status": status, "votes": votes_data}

def serialize_topic(t: Topic, agents_count: int, names: dict, db: Session) -> dict:
    status_info = compute_topic_status(t, agents_count, names)

    replies_data = []
    for r in t.replies:
        evals_data = [{"evaluator": names.get(e.evaluator_id, "Unknown"),
                       "score": e.score, "reason": e.reason, "timestamp": iso(e.created_at)}
                      for e in r.evaluations]
        replies_data.append({"id": r.id, "author": names.get(r.author_id, "Unknown"),
                             "content": r.content, "score": r.score, "timestamp": iso(r.created_at),
                             "evaluations": evals_data})

    linked_exps = db.query(Experiment).filter(Experiment.topic_id == t.id).all()
    linked_exp_data = [{
        "agent": names.get(exp.agent_id, "Unknown"),
        "method": exp.method, "cv_score": exp.cv_score, "lb_score": exp.lb_score,
        "notes": exp.notes, "timestamp": iso(exp.created_at)
    } for exp in linked_exps]

    return {
        "id": t.id, "creator": names.get(t.creator_id, "Unknown"),
        "title": t.title, "tag": t.tag, "content": t.content,
        "claimed_by": names.get(t.claimed_by_id) if t.claimed_by_id else None,
        "reply_count": len(t.replies), "timestamp": iso(t.created_at),
        "replies": replies_data, "status": status_info["status"], "votes": status_info["votes"],
        "linked_experiments": linked_exp_data,
        "conclusion": t.conclusion,
        "closed_by": names.get(t.closed_by_id) if t.closed_by_id else None,
        "closed_at": iso(t.closed_at) if t.closed_at else None
    }

def snippet(text_val: Optional[str], n: int = 80) -> str:
    if not text_val:
        return ""
    return text_val[:n] + ("..." if len(text_val) > n else "")

def compute_first_disagree(db: Session, p: Project) -> dict:
    """首个反对者: 项目内每个帖子按时间序第一条针对【他人提案】的【强烈反对】投票动作的归属者。
    三条铁律 (都是踩过坑的):
    1. 不能数 topic_votes 现票聚合——全员一致驳回机制会让每人的 disagree 现票数趋同, 没有区分度;
    2. 只认 ActivityLog 动作日志的时间序——现票是 UPSERT, created_at 是首次投票时间(发帖人
       的自动 verify 票天然最早), 用它回溯会把发帖人后来的改票错算成'第一个反对';
    3. 排除发帖人对自己提案的反对——那是'自我否决'(单独的指标), 不是率先批判。
    返回 {agent_id: [{topic_id, timestamp, description}, ...]}"""
    creators = {t.id: t.creator_id
                for t in db.query(Topic).filter(proj_crit(Topic.project_id, p)).all()}
    result = {}
    seen_topics = set()
    logs = (db.query(ActivityLog)
            .filter(proj_crit(ActivityLog.project_id, p))
            .filter(ActivityLog.action_type == "vote_topic",
                    ActivityLog.description.like("%强烈反对%"))
            .order_by(ActivityLog.id.asc()).all())
    for log in logs:
        tid = log.topic_id
        if tid is None:  # 旧日志无 topic_id 列, 从描述文本回退解析
            m = re.search(r"#(\d+)", log.description or "")
            tid = int(m.group(1)) if m else None
        if tid is None or tid in seen_topics or tid not in creators:
            continue
        if log.agent_id == creators.get(tid):
            continue  # 发帖人否决自己 != 率先批判他人
        seen_topics.add(tid)
        result.setdefault(log.agent_id, []).append(
            {"topic_id": tid, "timestamp": iso(log.created_at), "description": log.description})
    return result

def build_knowledge(p: Project, agents_count: int, names: dict, db: Session) -> list:
    """项目内已完结话题的结论清单: 全员共享的'哪些路已经走过'知识库。"""
    items = []
    for t in db.query(Topic).filter(proj_crit(Topic.project_id, p)).order_by(Topic.created_at.desc()).all():
        agree = sum(1 for v in t.votes if v.vote == "agree")
        disagree = sum(1 for v in t.votes if v.vote == "disagree")
        if t.closed_at is not None:
            outcome = "人工结案"
        elif agents_count > 0 and agree == agents_count:
            outcome = "通过"
        elif agents_count > 0 and disagree == agents_count:
            outcome = "驳回"
        else:
            continue  # 未完结
        items.append({
            "topic_id": t.id,
            "title": t.title,
            "tag": t.tag,
            "outcome": outcome,
            # 摘要而非全文: 完整结论用 GET /api/topics/{id}
            "conclusion": snippet(t.conclusion, 200),
            "closed_by": names.get(t.closed_by_id) if t.closed_by_id else None,
        })
    return items

def build_todo_items(agent: Agent, p: Project, agents_count: int, names: dict, db: Session) -> list:
    """项目内的状态性待办: 未投票 / 待跟进 / 待评分 / 待认领或执行中的任务。"""
    action_items = []
    now = datetime.utcnow()
    for t in db.query(Topic).filter(proj_crit(Topic.project_id, p)).order_by(Topic.created_at.desc()).all():
        reasons = []

        verify_count = sum(1 for v in t.votes if v.vote == "verify")
        agree_count = sum(1 for v in t.votes if v.vote == "agree")
        disagree_count = sum(1 for v in t.votes if v.vote == "disagree")
        is_resolved = (t.closed_at is not None) or (agree_count == agents_count) or (disagree_count == agents_count)
        is_todo = (not is_resolved) and (verify_count == agents_count)

        if not (is_resolved or is_todo):
            voted = any(v.agent_id == agent.id for v in t.votes)
            if not voted:
                reasons.append("尚未对该主题投票表态")

            sorted_replies = sorted(t.replies, key=lambda r: r.created_at)
            if len(sorted_replies) == 0:
                if t.creator_id != agent.id:
                    reasons.append("新主题，等待你的参与和表态")
            else:
                last_reply = sorted_replies[-1]
                if last_reply.author_id != agent.id:
                    reasons.append("有其他人的最新回复，等待你的跟进探讨")

            # 僵尸提案提示: 超 48h 未收敛, 提醒发起人主动收口
            if t.creator_id == agent.id and (now - t.created_at) > timedelta(hours=48):
                reasons.append("⏰ 你发起的该提案已讨论超 48 小时仍未收敛，请推动大家表态，或用 resolve 命令写下结论直接结案归档")
        elif is_resolved and t.closed_at is None and t.creator_id == agent.id:
            # 票数自然完结但没人写结论 -> 提醒发起人补写, 否则知识库里只有结局没有教训
            reasons.append("📚 【缺结论】你发起的该提案已完结但还没有结论文本，请用 resolve 命令补写一句结论 (验证了什么/结果如何/给后人的启示)，沉淀进知识库")
        elif is_todo:
            if not t.claimed_by_id:
                reasons.append("🚀 【全员通过】这是一个待执行的实验任务，请评估是否由你来认领 (使用 claim 命令)")
            else:
                delivered = db.query(Experiment).filter(
                    Experiment.topic_id == t.id, Experiment.agent_id == t.claimed_by_id).count() > 0
                if t.claimed_by_id == agent.id:
                    if delivered:
                        reasons.append("✅ 【已交付】你已提交实验结果，请在帖内回复结论并带头改投 agree/disagree，推动结案")
                    else:
                        reasons.append("⏳ 【开发中】你已认领该任务，请尽快编写代码跑出结果并用 experiment 命令记录 (记得带上 --topic_id)")
                elif delivered:
                    reasons.append("🔬 【实验已交付】认领人已提交实验结果，请查看结果并改投 agree/disagree 推动结案")

        # 人工归档的话题不再催评分 (归档即彻底封存); 票数自然完结的仍保留评分义务
        unevaluated_replies = []
        if t.closed_at is None:
            for r in t.replies:
                if r.author_id != agent.id:
                    has_eval = any(e.evaluator_id == agent.id for e in r.evaluations)
                    if not has_eval:
                        unevaluated_replies.append(r)

        if unevaluated_replies:
            reasons.append(f"有 {len(unevaluated_replies)} 条其他人的回复待你评分")

        if reasons:
            recent_replies = sorted(t.replies, key=lambda r: r.created_at, reverse=True)[:3]
            action_items.append({
                "topic_id": t.id,
                "title": t.title,
                "tag": t.tag,
                "creator": names.get(t.creator_id, "Unknown"),
                "reasons": reasons,
                "topic_preview": snippet(t.content, 150),
                # 摘要而非全文, 避免挤爆 AI 的上下文窗口; 全文用 GET /api/topics/{id}
                "recent_context": [
                    {"reply_id": r.id, "author": names.get(r.author_id, "Unknown"), "content": snippet(r.content, 200)}
                    for r in reversed(recent_replies)
                ],
                "pending_evaluations": [
                    {"reply_id": r.id, "author": names.get(r.author_id, "Unknown"), "content": snippet(r.content, 200)}
                    for r in unevaluated_replies
                ]
            })
    return action_items

# --- Routes ---

@app.get("/")
def read_root():
    return FileResponse(HUB_DIR / "static" / "index.html")

@app.get("/projects")
def projects_page():
    return FileResponse(HUB_DIR / "static" / "projects.html")

@app.get("/api/projects")
def list_projects(db: Session = Depends(get_db)):
    out = []
    for p in db.query(Project).order_by(Project.created_at.asc()).all():
        members = [a.name for _, a in project_members(db, p)]
        out.append({
            "name": p.name, "status": p.status,
            "metric_lower_is_better": p.metric_lower_is_better,
            "brief": p.brief,
            "member_count": len(members),
            "members": members,
            "topic_count": db.query(Topic).filter(proj_crit(Topic.project_id, p)).count(),
            "created_at": iso(p.created_at),
        })
    all_agents = [a.name for a in db.query(Agent).order_by(Agent.name.asc()).all()]
    return {"projects": out, "default_project": DEFAULT_PROJECT, "all_agents": all_agents}

@app.get("/api/system/status")
def system_status(db: Session = Depends(get_db)):
    projects = db.query(Project).count()
    agents = db.query(Agent).count()
    latest_log = db.query(func.max(ActivityLog.id)).scalar() or 0
    return {
        "status": "ok",
        "server_time": iso(datetime.utcnow()),
        "config": public_config(CONFIG),
        "database": {
            "connected": True,
            "projects": projects,
            "agents": agents,
            "latest_activity_id": latest_log,
        },
    }

@app.post("/api/projects/create")
def create_project(req: ProjectCreateReq, db: Session = Depends(get_db)):
    name = (req.name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,50}", name):
        raise HTTPException(status_code=400,
                            detail="项目名只允许字母/数字/下划线/连字符 (1-50 字符), 它会用于 URL 和 CLI 参数。")
    if db.query(Project).filter(Project.name == name).first():
        raise HTTPException(status_code=409, detail=f"项目 '{name}' 已存在。")
    p = Project(name=name, status="active",
                metric_lower_is_better=req.metric_lower_is_better,
                brief=(req.brief or "").strip() or None)
    db.add(p)
    db.flush()
    shared = ensure_project_workspace(p)
    db.add(ActivityLog(agent_id=None, action_type="project_created", project_id=p.id,
                       description=f"项目 '{p.name}' 已创建 (指标方向: {'越低越好' if p.metric_lower_is_better else '越高越好'}), 共享工作区: {shared}"))
    db.commit()
    return {"status": "success", "project": p.name, "workspace_shared": shared}

@app.post("/api/projects/{name}/update")
def update_project(name: str, req: ProjectUpdateReq, db: Session = Depends(get_db)):
    p = db.query(Project).filter(Project.name == name).first()
    if not p:
        raise HTTPException(status_code=404, detail=f"项目 '{name}' 不存在。")
    if req.status is not None:
        if req.status not in ("active", "archived"):
            raise HTTPException(status_code=400, detail="status 只能是 active 或 archived。")
        if req.status != p.status:
            p.status = req.status
            verb = "归档" if req.status == "archived" else "重新激活"
            db.add(ActivityLog(agent_id=None, action_type="project_status", project_id=p.id,
                               description=f"项目 '{p.name}' 已{verb}"))
    if req.brief is not None:
        p.brief = req.brief.strip() or None
    if req.metric_lower_is_better is not None:
        p.metric_lower_is_better = req.metric_lower_is_better
    db.commit()
    return {"status": "success", "project": p.name, "project_status": p.status}

class MemberAddReq(BaseModel):
    agent_name: str

@app.post("/api/projects/{name}/members")
def add_project_member(name: str, req: MemberAddReq, db: Session = Depends(get_db)):
    """人类从管理页把已注册的 AI 拉进项目 (不必等 AI 自己 update 加入)。"""
    p = resolve_project(db, name, for_write=True)
    agent = db.query(Agent).filter(Agent.name == req.agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_name}' 未注册过 (从未调用过 update)。")
    if get_state(db, agent, p):
        return {"status": "success", "already_member": True,
                "message": f"'{agent.name}' 已经是项目 '{p.name}' 的成员。"}
    state = get_state(db, agent, p, create=True)
    ws = ensure_agent_workspace(agent, p)
    state.workspace_dir = ws
    db.add(ActivityLog(agent_id=agent.id, action_type="join_project", project_id=p.id,
                       description=f"被拉入项目 '{p.name}', 私有工作区: {ws}"))
    db.commit()
    return {"status": "success", "already_member": False, "workspace": ws}

@app.post("/api/agents/update")
def update_agent(req: AgentUpdateReq, db: Session = Depends(get_db)):
    p = resolve_project(db, req.project, for_write=True)
    agent = db.query(Agent).filter(Agent.name == req.name).first()
    if agent is None:
        agent = Agent(name=req.name)
        db.add(agent)
        db.flush()  # 拿到 agent.id, 否则 ActivityLog 记到 System 头上

    state = get_state(db, agent, p)
    is_new_member = state is None
    if is_new_member:
        state = get_state(db, agent, p, create=True)
        # 自动创建私有工作区, 路径直接落到状态里 (AI 无需自报)
        ws = ensure_agent_workspace(agent, p)
        state.workspace_dir = ws
        db.add(ActivityLog(agent_id=agent.id, action_type="join_project", project_id=p.id,
                           description=f"加入了项目 '{p.name}', 私有工作区: {ws}"))

    status_changed = is_new_member or state.current_status != req.current_status
    state.current_status = req.current_status
    if req.workspace_dir is not None:
        state.workspace_dir = req.workspace_dir
    state.updated_at = datetime.utcnow()

    def record_score(score, score_type):
        # 仅在分数变化时落一条排行榜历史, 避免每轮汇报刷重复点
        last = (db.query(Leaderboard)
                .filter(Leaderboard.agent_name == req.name, Leaderboard.score_type == score_type,
                        proj_crit(Leaderboard.project_id, p))
                .order_by(Leaderboard.id.desc()).first())
        if last is None or last.score != score:
            db.add(Leaderboard(agent_name=req.name, score=score, score_type=score_type, project_id=p.id))

    if req.best_cv_score is not None:
        state.best_cv_score = req.best_cv_score
        record_score(req.best_cv_score, "cv")
    if req.best_lb_score is not None:
        state.best_lb_score = req.best_lb_score
        record_score(req.best_lb_score, "lb")

    if status_changed:
        db.add(ActivityLog(agent_id=agent.id, action_type="status_change", project_id=p.id,
                           description=f"更新了工作状态: {req.current_status}"))
    db.commit()
    return {"status": "success", "agent_id": agent.id, "project": p.name}

@app.post("/api/topics/create")
def create_topic(req: TopicCreateReq, db: Session = Depends(get_db)):
    p = resolve_project(db, req.project, for_write=True)
    creator = db.query(Agent).filter(Agent.name == req.creator_name).first()
    if not creator:
        raise HTTPException(status_code=404, detail=f"Agent '{req.creator_name}' not found. 请先用 update 命令注册。")
    require_member(db, creator, p)
    if not (req.tag and req.tag.strip()):
        raise HTTPException(status_code=400,
                            detail="发帖必须带分类标签 (--tag), 常用: 实验报告 / BUG修复 / 特征工程 / 模型融合 / 数据泄漏 / 日常交流。")
    topic = Topic(project_id=p.id, creator_id=creator.id, title=req.title, tag=req.tag, content=req.content)
    db.add(topic)
    db.flush()  # to get topic.id before commit
    # 发起人自动提议验证
    auto_vote = TopicVote(topic_id=topic.id, agent_id=creator.id, vote="verify", reason="发起人默认提议验证")
    db.add(auto_vote)

    db.add(ActivityLog(agent_id=creator.id, action_type="create_topic", topic_id=topic.id, project_id=p.id,
                       description=f"发起了提案 #{topic.id}【{topic.tag or '无标签'}】《{topic.title}》"))
    db.commit()
    return {"status": "success", "topic_id": topic.id, "project": p.name}

@app.post("/api/topics/reply")
def reply_to_topic(req: ReplyCreateReq, db: Session = Depends(get_db)):
    author = db.query(Agent).filter(Agent.name == req.author_name).first()
    if not author:
        raise HTTPException(status_code=404, detail=f"Agent '{req.author_name}' not found. 请先用 update 命令注册。")
    topic = db.query(Topic).get(req.topic_id)
    if not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{req.topic_id} 不存在: ID 可能写错或把回复编号当成了主题编号, 以 digest/read 输出的 #编号 为准。")
    p = project_of_topic(db, topic)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, author, p)
    reply = Reply(topic_id=topic.id, author_id=author.id, content=req.content, score=req.score)
    db.add(reply)
    db.flush()  # to get reply.id
    db.add(ActivityLog(agent_id=author.id, action_type="reply_topic", topic_id=topic.id, reply_id=reply.id,
                       project_id=p.id,
                       description=f"回复了提案 #{topic.id} (评分 {req.score}): {snippet(req.content)}"))
    db.commit()
    return {"status": "success", "reply_id": reply.id}

@app.post("/api/replies/evaluate")
def evaluate_reply(req: EvaluationReq, db: Session = Depends(get_db)):
    evaluator = db.query(Agent).filter(Agent.name == req.evaluator_name).first()
    if not evaluator:
        raise HTTPException(status_code=404, detail=f"Agent '{req.evaluator_name}' not found. 请先用 update 命令注册。")
    reply = db.query(Reply).get(req.reply_id)
    if not reply:
        raise HTTPException(status_code=404,
                            detail=f"回复 #{req.reply_id} 不存在: reply_id 是回复编号(read 输出里的 💬 回复 #N), 不是主题编号; 待评分回复的编号见 read 的'待评分的回复'列表。")
    topic = db.query(Topic).get(reply.topic_id)
    p = project_of_topic(db, topic)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, evaluator, p)
    ev = ReplyEvaluation(reply_id=reply.id, evaluator_id=evaluator.id, score=req.score, reason=req.reason)
    db.add(ev)
    author_name = agent_name_map(db).get(reply.author_id, "Unknown")
    db.add(ActivityLog(agent_id=evaluator.id, action_type="evaluate_reply",
                       topic_id=reply.topic_id, reply_id=reply.id, project_id=p.id,
                       description=f"给 {author_name} 在提案 #{reply.topic_id} 下的回复 #{reply.id} 评了 {req.score} 分: {snippet(req.reason)}"))
    db.commit()
    return {"status": "success", "evaluation_id": ev.id}

@app.post("/api/topics/vote")
def vote_on_topic(req: TopicVoteReq, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.name == req.agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_name}' not found. 请先用 update 命令注册。")
    topic = db.query(Topic).get(req.topic_id)
    if not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{req.topic_id} 不存在: 以 digest/read 输出的 #编号 为准。")
    if req.vote not in ("agree", "disagree", "verify"):
        raise HTTPException(status_code=400,
                            detail="vote 只能是 agree(赞成) / disagree(反对) / verify(逻辑成立, 但需要实验验证)。")
    p = project_of_topic(db, topic)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, agent, p)

    names = agent_name_map(db)
    agents_count = member_count(db, p)
    prev_status = compute_topic_status(topic, agents_count, names)["status"]

    existing = db.query(TopicVote).filter(TopicVote.topic_id == req.topic_id, TopicVote.agent_id == agent.id).first()
    if existing:
        existing.vote = req.vote
        existing.reason = req.reason
    else:
        db.add(TopicVote(topic_id=req.topic_id, agent_id=agent.id, vote=req.vote, reason=req.reason))

    v_str = "赞成" if req.vote == "agree" else "强烈反对" if req.vote == "disagree" else "必须实验验证"
    db.add(ActivityLog(agent_id=agent.id, action_type="vote_topic", topic_id=topic.id, project_id=p.id,
                       description=f"对提案 #{req.topic_id} 投出了【{v_str}】票: {snippet(req.reason)}"))

    # 仅在状态发生迁移时广播, 避免已完结的主题被反复公告
    db.flush()
    db.refresh(topic)
    new_status = compute_topic_status(topic, agents_count, names)["status"]
    if new_status != prev_status:
        if new_status == "已完结":
            db.add(ActivityLog(agent_id=None, action_type="topic_resolved", topic_id=topic.id, project_id=p.id,
                               description=f"提案 #{topic.id} 已被驳回/完结"))
        elif new_status == "待执行":
            db.add(ActivityLog(agent_id=None, action_type="topic_todo", topic_id=topic.id, project_id=p.id,
                               description=f"提案 #{topic.id} 已获全员通过，进入待执行状态"))

    db.commit()
    return {"status": "success", "topic_status": new_status}

@app.post("/api/experiment")
def log_experiment(req: ExperimentLogReq, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.name == req.agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_name}' not found. 请先用 update 命令注册。")
    if req.topic_id is None and not req.standalone:
        raise HTTPException(status_code=400,
                            detail="实验必须关联讨论话题 (--topic_id N), 否则结果无法回溯讨论背景。"
                                   "确实与任何话题无关的独立实验, 请显式加 --standalone。")
    topic = db.query(Topic).get(req.topic_id) if req.topic_id is not None else None
    if req.topic_id is not None and not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{req.topic_id} 不存在: 实验要挂载到已存在的讨论帖; 确实与任何话题无关时去掉 --topic_id 并加 --standalone。")
    # 项目作用域: 带 topic 从话题推导(忽略传入值, 杜绝不一致); standalone 用显式/默认项目
    p = project_of_topic(db, topic) if topic is not None else resolve_project(db, req.project, for_write=True)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, agent, p)

    exp = Experiment(project_id=p.id, agent_id=agent.id, topic_id=req.topic_id,
                     method=req.method, params=req.params,
                     cv_score=req.cv_score, lb_score=req.lb_score,
                     duration_min=req.duration_min, notes=req.notes)
    db.add(exp)

    desc_prefix = f"为提案 #{req.topic_id} " if req.topic_id else "独立"
    db.add(ActivityLog(agent_id=agent.id, action_type="log_experiment", topic_id=req.topic_id, project_id=p.id,
                       description=f"{desc_prefix}提交了实验结果 [{req.method}] CV:{req.cv_score} LB:{req.lb_score}"))

    # 认领任务的完成闭环: 认领人交付实验时广播, 提醒全员基于结果改投推动结案
    if topic is not None and topic.claimed_by_id == agent.id:
        db.add(ActivityLog(agent_id=None, action_type="topic_experiment_done", topic_id=topic.id, project_id=p.id,
                           description=f"提案 #{topic.id} 的认领实验已由 {agent.name} 交付结果，请各位查看并改投 agree/disagree 推动结案"))

    db.commit()
    return {"status": "success", "experiment_id": exp.id}

@app.post("/api/topics/claim")
def claim_topic(req: TopicClaimReq, db: Session = Depends(get_db)):
    agent = db.query(Agent).filter(Agent.name == req.agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_name}' not found. 请先用 update 命令注册。")
    topic = db.query(Topic).get(req.topic_id)
    if not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{req.topic_id} 不存在: 可认领的待执行任务以 read 待办 / digest 列出的 #编号 为准。")
    p = project_of_topic(db, topic)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, agent, p)
    if topic.claimed_by_id and topic.claimed_by_id != agent.id:
        owner = agent_name_map(db).get(topic.claimed_by_id, "Unknown")
        raise HTTPException(status_code=409,
                            detail=f"提案 #{topic.id} 已被 {owner} 认领，请不要重复跑同一个实验，去认领别的任务。")

    topic.claimed_by_id = agent.id
    db.add(ActivityLog(agent_id=agent.id, action_type="claim_topic", topic_id=topic.id, project_id=p.id,
                       description=f"主动认领了提案 #{req.topic_id} 的代码实验任务"))
    db.commit()
    return {"status": "success"}

@app.post("/api/topics/{topic_id}/resolve")
def resolve_topic(topic_id: int, req: TopicResolveReq, db: Session = Depends(get_db)):
    """人工结案: 不等票数收敛, 写下结论直接归档。结论会进入全员可见的知识库。"""
    agent = db.query(Agent).filter(Agent.name == req.agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_name}' not found. 请先用 update 命令注册。")
    topic = db.query(Topic).get(topic_id)
    if not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{topic_id} 不存在: 以 digest/read 输出的 #编号 为准。")
    p = project_of_topic(db, topic)
    if p.status == "archived":
        raise HTTPException(status_code=409, detail=f"项目 '{p.name}' 已归档, 只读不可写。")
    require_member(db, agent, p)
    if not req.conclusion or not req.conclusion.strip():
        raise HTTPException(status_code=400, detail="结案必须写结论 (--conclusion): 这是话题沉淀给全员的知识, 不能为空。")
    if topic.closed_at is not None:
        closer = agent_name_map(db).get(topic.closed_by_id, "Unknown")
        raise HTTPException(status_code=409, detail=f"提案 #{topic_id} 已由 {closer} 结案, 结论不可覆盖。如需补充请在帖内回复。")

    topic.closed_at = datetime.utcnow()
    topic.closed_by_id = agent.id
    topic.conclusion = req.conclusion.strip()
    db.add(ActivityLog(agent_id=agent.id, action_type="topic_resolved_manual", topic_id=topic.id, project_id=p.id,
                       description=f"人工结案了提案 #{topic.id}: {snippet(req.conclusion, 100)}"))
    db.commit()
    return {"status": "success", "topic_status": "已完结"}

@app.get("/api/read")
def read_updates(agent_name: str, mark_read: bool = True, project: Optional[str] = None,
                 db: Session = Depends(get_db)):
    """AI 收件箱(按项目): unread = 自上次读取以来本项目内别人产生的所有新事件;
    todo = 本项目的状态性待办。游标按 (agent, project) 独立维护。"""
    p = resolve_project(db, project)
    agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. 请先用 update 命令注册。")
    state = require_member(db, agent, p)

    names = agent_name_map(db)
    agents_count = member_count(db, p)
    cursor = state.last_read_log_id or 0

    rows = (db.query(ActivityLog)
            .filter(proj_crit(ActivityLog.project_id, p))
            .filter(ActivityLog.id > cursor)
            .filter(or_(ActivityLog.agent_id == None, ActivityLog.agent_id != agent.id))  # noqa: E711
            .order_by(ActivityLog.id.asc())
            .limit(UNREAD_PAGE_SIZE + 1)
            .all())
    truncated = len(rows) > UNREAD_PAGE_SIZE
    rows = rows[:UNREAD_PAGE_SIZE]

    topic_ids = {r.topic_id for r in rows if r.topic_id}
    topics = {t.id: t for t in db.query(Topic).filter(Topic.id.in_(topic_ids)).all()} if topic_ids else {}
    reply_ids = {r.reply_id for r in rows if r.reply_id}
    replies = {rp.id: rp for rp in db.query(Reply).filter(Reply.id.in_(reply_ids)).all()} if reply_ids else {}

    # 读取时已完结的话题: 其过程动态(发帖/回复/投票/结案)折叠为每话题一行,
    # 过程细节已无行动价值(评分义务在 todo 里单独带摘要), 需要回看用 get
    closed_now = {tid: compute_topic_status(t, agents_count, names)["status"] == "已完结"
                  for tid, t in topics.items()}
    action_labels = {"create_topic": "发帖", "reply_topic": "回复", "vote_topic": "投票",
                     "evaluate_reply": "评分", "log_experiment": "实验", "claim_topic": "认领",
                     "topic_resolved": "票决完结", "topic_resolved_manual": "人工结案",
                     "topic_todo": "转待执行", "topic_experiment_done": "交付广播",
                     "status_change": "状态更新", "join_project": "加入项目"}

    events = []
    bundles = {}  # topic_id -> 折叠事件 (保持首次出现的时间序位置)
    for r in rows:
        actor = names.get(r.agent_id, "Unknown") if r.agent_id else "System"
        if r.topic_id and closed_now.get(r.topic_id):
            b = bundles.get(r.topic_id)
            if b is None:
                t = topics[r.topic_id]
                b = {"type": "closed_topic_bundle",
                     "topic": {"id": t.id, "title": t.title, "tag": t.tag},
                     "count": 0, "actions": {}, "actors": [], "timestamp": iso(r.created_at)}
                bundles[r.topic_id] = b
                events.append(b)
            b["count"] += 1
            label = action_labels.get(r.action_type, r.action_type)
            b["actions"][label] = b["actions"].get(label, 0) + 1
            if actor not in b["actors"]:
                b["actors"].append(actor)
            b["timestamp"] = iso(r.created_at)
            continue
        ev = {
            "id": r.id,
            "type": r.action_type,
            "actor": actor,
            "description": r.description,
            "timestamp": iso(r.created_at),
        }
        if r.topic_id and r.topic_id in topics:
            t = topics[r.topic_id]
            ev["topic"] = {"id": t.id, "title": t.title, "tag": t.tag}
        if r.reply_id and r.reply_id in replies and r.action_type == "reply_topic":
            rp = replies[r.reply_id]
            # 摘要而非全文(全文用 GET /api/topics/{id}); 评分等事件不重复附带回复原文——
            # 该回复创建时已作为独立事件出现过, description 里自带评分理由
            ev["reply"] = {"id": rp.id, "author": names.get(rp.author_id, "Unknown"),
                           "content": snippet(rp.content, 200), "score": rp.score}
        events.append(ev)

    # 推进游标: 未截断时可越过自己的事件直达本项目最新; 截断时只推进到已返回的位置, 不丢事件
    if truncated:
        new_cursor = rows[-1].id
    else:
        new_cursor = (db.query(func.max(ActivityLog.id))
                      .filter(proj_crit(ActivityLog.project_id, p)).scalar()) or cursor
    if mark_read and new_cursor > cursor:
        state.last_read_log_id = new_cursor
        db.commit()

    todo = build_todo_items(agent, p, agents_count, names, db)

    return {
        "agent": agent.name,
        "project": p.name,
        "unread": {"count": len(events), "truncated": truncated, "events": events},
        "todo": todo,
        # knowledge(已结案结论)只随 digest 下发, read 不再重复携带 (省 AI 上下文)
        "cursor": new_cursor,
        # 兼容旧客户端字段
        "action_items": todo,
    }

@app.get("/api/digest")
def get_digest(agent_name: str, project: Optional[str] = None, db: Session = Depends(get_db)):
    """项目态势摘要: AI 进入项目时一次调用对齐大局, 这是冷启动包的核心。"""
    p = resolve_project(db, project)
    agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. 请先用 update 命令注册。")
    state = get_state(db, agent, p)  # 非成员也允许看 digest (入伙前先了解项目)

    names = agent_name_map(db)
    members = project_members(db, p)
    agents_count = len(members)
    member_ids = {a.id for _, a in members}
    now = datetime.utcnow()
    day_ago = now - timedelta(hours=24)

    recent_topics = []       # 近 24h 新建且仍未完结 (已完结的合并进 recent_closed_24h, 结论看知识库)
    recent_closed_ids = []   # 近 24h 新建且已完结
    near_consensus = []      # 差 1 票即达成共识
    missing_conclusions = [] # 已完结但缺结论, 等发起人 resolve 补写
    open_count = 0
    for t in db.query(Topic).filter(proj_crit(Topic.project_id, p)).order_by(Topic.created_at.desc()).all():
        status_info = compute_topic_status(t, agents_count, names)
        status = status_info["status"]
        if t.created_at >= day_ago:
            if status == "已完结":
                recent_closed_ids.append(t.id)
            else:
                recent_topics.append({"topic_id": t.id, "title": t.title, "tag": t.tag,
                                      "status": status, "creator": names.get(t.creator_id, "Unknown")})
        if status == "已完结":
            if t.closed_at is None:
                missing_conclusions.append({"topic_id": t.id, "title": t.title,
                                            "creator": names.get(t.creator_id, "Unknown")})
            continue
        open_count += 1
        # 差 1 票分析: 某一票型只差 1 人即触发状态迁移 (只看项目成员)
        votes_by_agent = {v.agent_id: v.vote for v in t.votes}
        for kind, label in (("verify", "全员verify→待执行"), ("agree", "全员agree→完结"), ("disagree", "全员disagree→驳回")):
            count = sum(1 for aid in member_ids if votes_by_agent.get(aid) == kind)
            if agents_count > 1 and count == agents_count - 1:
                holdouts = [names[aid] for aid in member_ids if votes_by_agent.get(aid) != kind]
                near_consensus.append({"topic_id": t.id, "title": t.title, "missing": label,
                                       "holdouts": holdouts, "you_are_holdout": agent.name in holdouts})

    recent_exps = []
    for exp in (db.query(Experiment).filter(proj_crit(Experiment.project_id, p))
                .order_by(Experiment.created_at.desc()).limit(5).all()):
        recent_exps.append({"agent": names.get(exp.agent_id, "Unknown"), "topic_id": exp.topic_id,
                            "method": exp.method, "cv_score": exp.cv_score, "lb_score": exp.lb_score,
                            "notes": snippet(exp.notes, 120), "timestamp": iso(exp.created_at)})

    agents_data = [{"name": a.name, "status": snippet(st.current_status, 200),
                    "cv_score": st.best_cv_score, "lb_score": st.best_lb_score,
                    "updated_at": iso(st.updated_at) if st.updated_at else None}
                   for st, a in members]

    unread_count = 0
    if state:
        unread_count = (db.query(func.count(ActivityLog.id))
                        .filter(proj_crit(ActivityLog.project_id, p))
                        .filter(ActivityLog.id > (state.last_read_log_id or 0))
                        .filter(or_(ActivityLog.agent_id == None, ActivityLog.agent_id != agent.id))  # noqa: E711
                        .scalar())
    todo_count = len(build_todo_items(agent, p, agents_count, names, db)) if state else 0

    return {
        "agent": agent.name,
        "project": {
            "name": p.name,
            "status": p.status,
            "metric_lower_is_better": p.metric_lower_is_better,
            "brief": p.brief,
            # 工作区路径由平台给出, AI 不要靠猜: 共享层放正典/共享产物, 私有区放实验代码和 NOTES.md
            "workspace_shared": str(WORKSPACE_ROOT / p.name),
            "my_workspace": abs_workspace(state.workspace_dir) if state else None,
        },
        "is_member": state is not None,
        "agents": agents_data,
        "open_topics": open_count,
        "recent_topics_24h": recent_topics,
        "recent_closed_24h": {"count": len(recent_closed_ids), "ids": recent_closed_ids},
        "near_consensus": near_consensus,
        "recent_experiments": recent_exps,
        "knowledge": build_knowledge(p, agents_count, names, db),
        "missing_conclusions": missing_conclusions,
        "my_unread_count": unread_count,
        "my_todo_count": todo_count,
    }

@app.get("/api/agents/{agent_name}/drilldown")
def agent_drilldown(agent_name: str, metric: str, project: Optional[str] = None,
                    db: Session = Depends(get_db)):
    """成员卡指标的明细下钻(按项目): 点数字看逐条证据, 每条可跳回所在帖子。"""
    p = resolve_project(db, project)
    agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found.")
    names = agent_name_map(db)
    proj_topics = db.query(Topic).filter(proj_crit(Topic.project_id, p)).all()
    topic_titles = {t.id: t.title for t in proj_topics}
    proj_replies = (db.query(Reply.id).join(Topic, Reply.topic_id == Topic.id)
                    .filter(proj_crit(Topic.project_id, p)).subquery())

    def eval_items(q, counterpart_of):
        out = []
        for ev, rp in q.order_by(ReplyEvaluation.created_at.desc()).all():
            out.append({
                "topic_id": rp.topic_id,
                "topic_title": topic_titles.get(rp.topic_id, f"#{rp.topic_id}"),
                "reply_id": rp.id,
                "reply_snippet": snippet(rp.content, 120),
                "score": ev.score,
                "reason": ev.reason,
                "counterpart": names.get(counterpart_of(ev, rp), "Unknown"),
                "timestamp": iso(ev.created_at),
            })
        return out

    base = (db.query(ReplyEvaluation, Reply)
            .join(Reply, ReplyEvaluation.reply_id == Reply.id)
            .filter(Reply.id.in_(proj_replies)))
    if metric == "evals_received":
        title = "收到的全部评价"
        items = eval_items(base.filter(Reply.author_id == agent.id), lambda ev, rp: ev.evaluator_id)
    elif metric == "low_evals_received":
        title = "收到的差评 (<5 分)"
        items = eval_items(base.filter(Reply.author_id == agent.id, ReplyEvaluation.score < 5),
                           lambda ev, rp: ev.evaluator_id)
    elif metric == "harsh_evals_given":
        title = "打出的差评 (<5 分)"
        items = eval_items(base.filter(ReplyEvaluation.evaluator_id == agent.id, ReplyEvaluation.score < 5),
                           lambda ev, rp: rp.author_id)
    elif metric == "first_disagree":
        title = "率先反对他人提案的记录"
        items = [{
            "topic_id": rec["topic_id"],
            "topic_title": topic_titles.get(rec["topic_id"], f"#{rec['topic_id']}"),
            "reply_id": None, "reply_snippet": None, "score": None, "counterpart": None,
            "reason": rec["description"],
            "timestamp": rec["timestamp"],
        } for rec in compute_first_disagree(db, p).get(agent.id, [])]
    elif metric == "todo":
        title = "待办事项"
        agents_count = member_count(db, p)
        items = [{
            "topic_id": it["topic_id"],
            "topic_title": it["title"],
            "reply_id": None, "reply_snippet": None, "score": None,
            "counterpart": it["creator"],
            "reason": "；".join(it["reasons"]),
            "timestamp": None,
        } for it in build_todo_items(agent, p, agents_count, names, db)]
    elif metric == "self_rejections":
        title = "自我否决的提案 (被说服后亲手投反对)"
        items = []
        for t in proj_topics:
            if t.creator_id != agent.id:
                continue
            own_vote = next((v for v in t.votes if v.agent_id == agent.id and v.vote == "disagree"), None)
            if own_vote:
                items.append({
                    "topic_id": t.id,
                    "topic_title": t.title,
                    "reply_id": None, "reply_snippet": None, "score": None, "counterpart": None,
                    "reason": own_vote.reason,
                    "timestamp": iso(own_vote.created_at),
                })
        items.sort(key=lambda x: x["timestamp"], reverse=True)
    else:
        raise HTTPException(status_code=400,
                            detail="metric 必须是 evals_received / low_evals_received / harsh_evals_given / first_disagree / self_rejections / todo 之一")

    return {"agent": agent.name, "metric": metric, "title": title, "items": items, "project": p.name}

@app.get("/api/topics/{topic_id}")
def get_topic_detail(topic_id: int, db: Session = Depends(get_db)):
    topic = db.query(Topic).get(topic_id)
    if not topic:
        raise HTTPException(status_code=404,
                            detail=f"主题 #{topic_id} 不存在: 以 digest/read 输出的 #编号 为准。")
    p = project_of_topic(db, topic)
    names = agent_name_map(db)
    return serialize_topic(topic, member_count(db, p), names, db)

@app.get("/api/dashboard_data")
def get_dashboard_data(project: Optional[str] = None, db: Session = Depends(get_db)):
    p = resolve_project(db, project)
    names = agent_name_map(db)
    members = project_members(db, p)
    agents_count = len(members)

    # Leaderboard
    lb_data = []
    for lb in (db.query(Leaderboard).filter(proj_crit(Leaderboard.project_id, p))
               .order_by(Leaderboard.created_at.asc()).all()):
        lb_data.append({"agent_name": lb.agent_name, "score": lb.score,
                        "score_type": lb.score_type or "cv",
                        "timestamp": iso(lb.created_at)})

    # Topics
    topics = (db.query(Topic).filter(proj_crit(Topic.project_id, p))
              .order_by(Topic.created_at.desc()).all())
    all_tags = set()
    topics_data = []
    for t in topics:
        if t.tag:
            all_tags.add(t.tag)
        topics_data.append(serialize_topic(t, agents_count, names, db))

    # Experiments
    experiments = (db.query(Experiment).filter(proj_crit(Experiment.project_id, p))
                   .order_by(Experiment.created_at.desc()).limit(50).all())
    exp_data = []
    for exp in experiments:
        exp_data.append({
            "agent": names.get(exp.agent_id, "Unknown"), "method": exp.method,
            "params": exp.params, "cv_score": exp.cv_score, "lb_score": exp.lb_score,
            "duration_min": exp.duration_min, "notes": exp.notes,
            "timestamp": iso(exp.created_at)
        })

    first_disagree_map = compute_first_disagree(db, p)
    proj_topic_ids = [t.id for t in topics]
    proj_replies = (db.query(Reply.id).filter(Reply.topic_id.in_(proj_topic_ids)).subquery()
                    if proj_topic_ids else None)

    agents_data = []
    for st, a in members:
        # ---- 提案维度 (作为发起人, 项目内) ----
        agent_topics = [t for t in topics if t.creator_id == a.id]
        topic_count = len(agent_topics)
        approved_count = 0
        rejected_count = 0
        archived_count = 0
        voting_count = 0
        disagree_received = 0  # 提案累计收到的反对票 (含未完结的, 软驳回信号)
        self_reject_count = 0  # 自我否决: 被说服后亲手对自己的提案投反对 (论坛文化里的美德)
        for t in agent_topics:
            agree = sum(1 for v in t.votes if v.vote == "agree")
            disagree = sum(1 for v in t.votes if v.vote == "disagree")
            verify = sum(1 for v in t.votes if v.vote == "verify")
            disagree_received += disagree
            if any(v.agent_id == a.id and v.vote == "disagree" for v in t.votes):
                self_reject_count += 1

            if verify == agents_count or agree == agents_count:
                approved_count += 1
            elif disagree == agents_count:
                rejected_count += 1
            elif t.closed_at is not None:
                archived_count += 1
            else:
                voting_count += 1

        reply_count = (db.query(Reply).filter(Reply.author_id == a.id,
                                              Reply.id.in_(proj_replies)).count()
                       if proj_replies is not None else 0)
        claimed_count = sum(1 for t in topics if t.claimed_by_id == a.id)

        # ---- 发言维度 (作为回复者, 项目内): 收到的同行评分 ----
        avg_eval_score = None
        low_eval_count = 0
        harsh_evals_given = 0
        if proj_replies is not None:
            my_replies = db.query(Reply.id).filter(Reply.author_id == a.id,
                                                   Reply.id.in_(proj_replies)).subquery()
            avg_score_res = db.query(func.avg(ReplyEvaluation.score)).filter(
                ReplyEvaluation.reply_id.in_(my_replies)).scalar()
            avg_eval_score = float(avg_score_res) if avg_score_res is not None else None
            low_eval_count = db.query(func.count(ReplyEvaluation.id)).filter(
                ReplyEvaluation.reply_id.in_(my_replies), ReplyEvaluation.score < 5).scalar() or 0
            harsh_evals_given = db.query(func.count(ReplyEvaluation.id)).filter(
                ReplyEvaluation.evaluator_id == a.id,
                ReplyEvaluation.reply_id.in_(proj_replies), ReplyEvaluation.score < 5).scalar() or 0

        # ---- 待办与未读 (供人类决定唤醒哪个 AI 处理积压) ----
        todo_count = len(build_todo_items(a, p, agents_count, names, db))
        unread_count = (db.query(func.count(ActivityLog.id))
                        .filter(proj_crit(ActivityLog.project_id, p))
                        .filter(ActivityLog.id > (st.last_read_log_id or 0))
                        .filter(or_(ActivityLog.agent_id == None, ActivityLog.agent_id != a.id))  # noqa: E711
                        .scalar() or 0)

        # 率先反对: 项目内在帖子上第一个对他人提案投反对的次数
        first_disagree_count = len(first_disagree_map.get(a.id, []))

        agents_data.append({
            "name": a.name, "status": st.current_status, "cv_score": st.best_cv_score,
            "lb_score": st.best_lb_score, "workspace": st.workspace_dir,
            "updated_at": iso(st.updated_at) if st.updated_at else None,
            "topic_count": topic_count,
            "reply_count": reply_count,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "archived_count": archived_count,
            "voting_count": voting_count,
            "claimed_count": claimed_count,
            "avg_eval_score": avg_eval_score,
            "disagree_received": disagree_received,
            "low_eval_count": low_eval_count,
            "first_disagree_count": first_disagree_count,
            "harsh_evals_given": harsh_evals_given,
            "self_reject_count": self_reject_count,
            "todo_count": todo_count,
            "unread_count": unread_count
        })

    # Activity Feed
    logs = (db.query(ActivityLog).filter(proj_crit(ActivityLog.project_id, p))
            .order_by(ActivityLog.created_at.desc()).limit(25).all())
    activity_feed = []
    for log in logs:
        agent_name = "System"
        if log.agent_id:
            agent_name = names.get(log.agent_id, "Unknown")
        activity_feed.append({
            "type": log.action_type,
            "agent": agent_name,
            "desc": log.description,
            "timestamp": iso(log.created_at)
        })

    return {
        "current_project": p.name,
        "project_status": p.status,
        "agents": agents_data,
        "topics": topics_data, "leaderboard": lb_data, "experiments": exp_data,
        "all_tags": sorted(list(all_tags)),
        "activity_feed": activity_feed,
        "metric_lower_is_better": p.metric_lower_is_better
    }

@app.get("/api/project_plugin/{project}/{action}")
def project_plugin_action(project: str, action: str, req: fastapi.Request, db: Session = Depends(get_db)):
    import importlib.util
    import sys
    import os
    plugin_dir = HUB_DIR / "plugins" / project
    api_file = plugin_dir / "api.py"
    if not api_file.exists():
        raise HTTPException(status_code=404, detail=f"Plugin API not found for project '{project}'")
    
    module_name = f"plugins.{project}.api"
    spec = importlib.util.spec_from_file_location(module_name, str(api_file))
    if spec is None or spec.loader is None:
        raise HTTPException(status_code=500, detail="Failed to load plugin spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing plugin: {e}")
    
    if not hasattr(module, "handle_request"):
        raise HTTPException(status_code=500, detail="Plugin API must define 'handle_request(action, request, db)'")
    
    return module.handle_request(action, req, db)
