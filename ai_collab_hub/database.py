import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, UniqueConstraint, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
try:
    from .config import load_config
except ImportError:
    from config import load_config

CONFIG = load_config()
DATABASE_URL = CONFIG["database"]["url"]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)  # URL/CLI 中使用的短名, 如 'rogii'
    status = Column(String(20), nullable=False, default="active")  # 'active' | 'archived'
    metric_lower_is_better = Column(Boolean, nullable=False, default=True)  # RMSE类=True, AUC类=False
    brief = Column(Text, nullable=True)  # 项目简报: AI 冷启动指针页, digest 原样返回
    created_at = Column(DateTime, default=datetime.utcnow)

class Agent(Base):
    """全局 AI 身份注册表。项目级状态(状态/分数/游标/工作区)在 AgentProjectState;
    本表同名旧列已冻结不再读写, 仅为代码回滚保留。"""
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, index=True)
    current_status = Column(String(255))
    best_cv_score = Column(Float, nullable=True)
    best_lb_score = Column(Float, nullable=True)
    workspace_dir = Column(String(255))
    # 未读游标: 指向该 agent 已读到的 activity_logs.id
    last_read_log_id = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AgentProjectState(Base):
    """AI 在某项目内的状态。行的存在即成员资格(共识分母按此计);
    未读游标必须按 (agent, project) 拆分 -- 单游标在项目 A read 会推过项目 B 的未读事件。"""
    __tablename__ = "agent_project_state"
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    current_status = Column(String(255), nullable=True)
    best_cv_score = Column(Float, nullable=True)
    best_lb_score = Column(Float, nullable=True)
    last_read_log_id = Column(Integer, nullable=False, default=0)
    workspace_dir = Column(String(255), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    agent = relationship("Agent")
    project = relationship("Project")
    __table_args__ = (UniqueConstraint("agent_id", "project_id", name="uq_agent_project"),)

class Topic(Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)  # 应用层视为必填
    creator_id = Column(Integer, ForeignKey("agents.id"))
    title = Column(String(255))
    tag = Column(String(50), nullable=True)  # 分类标签
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    claimed_by_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    # 人工结案: 不再等票数收敛, 由任一 agent 写结论直接归档
    closed_at = Column(DateTime, nullable=True)
    closed_by_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    conclusion = Column(Text, nullable=True)
    creator = relationship("Agent", foreign_keys=[creator_id])
    claimed_by = relationship("Agent", foreign_keys=[claimed_by_id])
    closed_by = relationship("Agent", foreign_keys=[closed_by_id])
    replies = relationship("Reply", back_populates="topic", cascade="all, delete-orphan", order_by="Reply.created_at")
    votes = relationship("TopicVote", back_populates="topic", cascade="all, delete-orphan")

class Reply(Base):
    __tablename__ = "replies"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    author_id = Column(Integer, ForeignKey("agents.id"))
    content = Column(Text)
    score = Column(Float, nullable=False, default=0.0)  # 强制要求的评分
    created_at = Column(DateTime, default=datetime.utcnow)
    topic = relationship("Topic", back_populates="replies")
    author = relationship("Agent")
    evaluations = relationship("ReplyEvaluation", back_populates="reply", cascade="all, delete-orphan")

class ReplyEvaluation(Base):
    __tablename__ = "reply_evaluations"
    id = Column(Integer, primary_key=True, index=True)
    reply_id = Column(Integer, ForeignKey("replies.id"))
    evaluator_id = Column(Integer, ForeignKey("agents.id"))
    score = Column(Float)
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    reply = relationship("Reply", back_populates="evaluations")
    evaluator = relationship("Agent")

class TopicVote(Base):
    __tablename__ = "topic_votes"
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    agent_id = Column(Integer, ForeignKey("agents.id"))
    vote = Column(String(20))
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    topic = relationship("Topic", back_populates="votes")
    agent = relationship("Agent")

class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)  # standalone 实验无 topic 可推导, 必须自带
    agent_id = Column(Integer, ForeignKey("agents.id"))
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    method = Column(String(255))
    params = Column(Text, nullable=True)
    cv_score = Column(Float, nullable=True)
    lb_score = Column(Float, nullable=True)
    duration_min = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    agent = relationship("Agent")
    topic = relationship("Topic")

class ActivityLog(Base):
    __tablename__ = "activity_logs"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)  # read 按项目过滤的关键列
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    action_type = Column(String(50))
    description = Column(Text)
    # 事件关联的对象引用, 供未读收件箱携带上下文
    topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    reply_id = Column(Integer, ForeignKey("replies.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    agent = relationship("Agent")

class Leaderboard(Base):
    __tablename__ = "leaderboard_history"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    agent_name = Column(String(50), index=True)
    score = Column(Float)
    score_type = Column(String(10), default="cv")  # "cv" or "lb"
    created_at = Column(DateTime, default=datetime.utcnow)

class NeuroGolfArtifact(Base):
    __tablename__ = "neurogolf_artifacts"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    task_id = Column(String(20), nullable=False, index=True)
    score = Column(Float, nullable=True)
    verified_status = Column(String(40), nullable=False, default="UNKNOWN")
    sha256 = Column(String(64), nullable=False)
    bytes = Column(Integer, nullable=False, default=0)
    forum_topic_id = Column(Integer, ForeignKey("topics.id"), nullable=True)
    created_by = Column(String(50), nullable=True)
    artifact_path = Column(String(512), nullable=False)
    is_deployed = Column(Boolean, nullable=False, default=False)
    is_dummy = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class KaggleSubmission(Base):
    __tablename__ = "kaggle_submissions"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    message = Column(String(500), nullable=True)
    public_score = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)
    total_teams = Column(Integer, nullable=True)
    status = Column(String(50), default="pending")  # pending / complete / error
    solved_count = Column(Integer, nullable=True)
    submitted_by = Column(String(100), nullable=True)
    kaggle_ref = Column(String(40), nullable=True, index=True)  # Kaggle submission id, for reconcile dedup

Base.metadata.create_all(bind=engine)

def _migrate(engine):
    """为已存在的库补齐新列 (create_all 不会 ALTER 已有表)。"""
    insp = inspect(engine)
    with engine.begin() as conn:
        agent_cols = {c["name"] for c in insp.get_columns("agents")}
        if "last_read_log_id" not in agent_cols:
            conn.execute(text("ALTER TABLE agents ADD COLUMN last_read_log_id INT NOT NULL DEFAULT 0"))
            # 已有 agent 的游标对齐到当前最新日志, 避免历史事件全量倾倒为未读
            conn.execute(text("UPDATE agents SET last_read_log_id = (SELECT COALESCE(MAX(id), 0) FROM activity_logs)"))
        log_cols = {c["name"] for c in insp.get_columns("activity_logs")}
        if "topic_id" not in log_cols:
            conn.execute(text("ALTER TABLE activity_logs ADD COLUMN topic_id INT NULL"))
        if "reply_id" not in log_cols:
            conn.execute(text("ALTER TABLE activity_logs ADD COLUMN reply_id INT NULL"))
        topic_cols = {c["name"] for c in insp.get_columns("topics")}
        if "closed_at" not in topic_cols:
            conn.execute(text("ALTER TABLE topics ADD COLUMN closed_at DATETIME NULL"))
        if "closed_by_id" not in topic_cols:
            conn.execute(text("ALTER TABLE topics ADD COLUMN closed_by_id INT NULL"))
        if "conclusion" not in topic_cols:
            conn.execute(text("ALTER TABLE topics ADD COLUMN conclusion TEXT NULL"))
        if insp.has_table("kaggle_submissions"):
            sub_cols = {c["name"] for c in insp.get_columns("kaggle_submissions")}
            if "kaggle_ref" not in sub_cols:
                conn.execute(text("ALTER TABLE kaggle_submissions ADD COLUMN kaggle_ref VARCHAR(40) NULL"))

        # ---- 多项目迁移: 现有数据全部归入默认项目 'rogii' ----
        # projects / agent_project_state 两张新表由上面的 create_all 建出
        if conn.execute(text("SELECT COUNT(*) FROM projects")).scalar() == 0:
            conn.execute(text("INSERT INTO projects (name, status, metric_lower_is_better, created_at) "
                              "VALUES ('rogii', 'active', 1, NOW())"))
        default_pid = conn.execute(text("SELECT id FROM projects ORDER BY id ASC LIMIT 1")).scalar()

        for tbl in ("topics", "experiments", "activity_logs", "leaderboard_history"):
            cols = {c["name"] for c in insp.get_columns(tbl)}
            if "project_id" not in cols:
                conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN project_id INT NULL"))
            conn.execute(text(f"UPDATE {tbl} SET project_id = :pid WHERE project_id IS NULL"),
                         {"pid": default_pid})

        # agents 的项目级状态(状态/分数/游标/工作区)原值平移进 agent_project_state,
        # 3 个 AI 自动成为 rogii 成员; 游标平移保证不产生未读洪水
        if conn.execute(text("SELECT COUNT(*) FROM agent_project_state")).scalar() == 0 \
                and conn.execute(text("SELECT COUNT(*) FROM agents")).scalar() > 0:
            conn.execute(text(
                "INSERT INTO agent_project_state (agent_id, project_id, current_status, best_cv_score, "
                "best_lb_score, last_read_log_id, workspace_dir, updated_at) "
                "SELECT id, :pid, current_status, best_cv_score, best_lb_score, "
                "COALESCE(last_read_log_id, 0), workspace_dir, updated_at FROM agents"),
                {"pid": default_pid})

        artifact_cols = {c["name"] for c in insp.get_columns("neurogolf_artifacts")} \
            if "neurogolf_artifacts" in insp.get_table_names() else set()
        # create_all creates the table for fresh databases. These guards keep older
        # partially migrated central hubs compatible if a column is added later.
        expected_cols = {
            "project_id": "INT NOT NULL",
            "task_id": "VARCHAR(20) NOT NULL",
            "score": "FLOAT NULL",
            "verified_status": "VARCHAR(40) NOT NULL DEFAULT 'UNKNOWN'",
            "sha256": "VARCHAR(64) NOT NULL",
            "bytes": "INT NOT NULL DEFAULT 0",
            "forum_topic_id": "INT NULL",
            "created_by": "VARCHAR(50) NULL",
            "artifact_path": "VARCHAR(512) NOT NULL",
            "is_deployed": "BOOL NOT NULL DEFAULT 0",
            "is_dummy": "BOOL NOT NULL DEFAULT 0",
            "created_at": "DATETIME NULL",
            "updated_at": "DATETIME NULL",
        }
        for col, ddl in expected_cols.items():
            if artifact_cols and col not in artifact_cols:
                conn.execute(text(f"ALTER TABLE neurogolf_artifacts ADD COLUMN {col} {ddl}"))

_migrate(engine)
