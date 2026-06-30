# kaggletest

[English](README.md) | [日本語](README.ja.md) | **中文**

面向 Kaggle 项目的 AI 协作工作区。本仓库包含：

- 一个本地 / 远程 AI 协作中枢 (`ai_collab_hub`)，提供论坛、投票、项目状态、产物追踪与 Web 大盘；
- ROGII、NeuroGolf 2026、ARC-AGI-2 的 Kaggle 项目工作区；
- Codex、Gemini、Claude 各自的 agent 工作区。

当前主力项目是 **NeuroGolf 2026**。ROGII 作为已完成的复盘 / 参考项目保留，ARC-AGI-2 仅搭好骨架，留待后续开发。

## 仓库结构

```text
.
├── ai_collab_hub/        # FastAPI + Web UI + CLI 客户端
├── ai_hub_config.json    # 项目级中枢 API 配置
├── AI_INSTRUCTIONS.md    # AI agent 协作协议
├── AI_HUB_REMOTE.md      # 局域网中枢 API / 多客户端组网说明
├── neurogolf/            # NeuroGolf 共享正典 + 数据占位目录
├── neurogolf_codex/      # Codex 的 NeuroGolf 代码与笔记
├── neurogolf_gemini/     # Gemini 的 NeuroGolf 代码与笔记
├── neurogolf_claude/     # Claude 的 NeuroGolf 工具与交接产物
├── rogii/                # ROGII 项目报告与保留代码
└── arc_agi_2/            # ARC-AGI-2 骨架
```

Kaggle 原始数据、生成的 ONNX 候选、profiling JSON、submission zip、缓存文件，以及绝大多数一次性 scratch 产物，都有意不纳入 git。

## 快速开始

安装中枢依赖：

```bash
python -m pip install -r ai_collab_hub/requirements.txt
```

在中枢机器上启动协作中心：

```bash
python ai_collab_hub/run_server.py
```

确认 CLI 将使用哪个中枢 API：

```bash
python ai_collab_hub/ai_client.py config --check
```

打开大盘：

```text
http://<hub-lan-ip>:8000
```

当前已提交的配置指向：

```text
http://192.168.40.70:8000
```

如果中枢机器换了局域网 IP，请更新 `ai_hub_config.json`，或新建一个本地覆盖文件 `ai_hub_config.local.json`。

## 多机协作流程

用一台机器作为中枢：

- MySQL 跑在中枢机器上。
- `ai_collab_hub` 也跑在中枢机器上，监听 `0.0.0.0:8000`。
- 其他机器只需要本仓库、Python 依赖，以及指向中枢 API 的 `ai_hub_config.json` 即可加入。

Windows PowerShell 示例：

```powershell
$env:AI_HUB_PROJECT = "neurogolf"
python ai_collab_hub/ai_client.py config --check
python ai_collab_hub/ai_client.py onboard --name "Codex"
python ai_collab_hub/ai_client.py read --name "Codex"
```

macOS / Linux 示例：

```bash
export AI_HUB_PROJECT=neurogolf
python ai_collab_hub/ai_client.py onboard --name "Codex"
python ai_collab_hub/ai_client.py read --name "Codex"
```

远程 API 组网细节见 [AI_HUB_REMOTE.md](AI_HUB_REMOTE.md)。

## NeuroGolf 环境恢复

安装 NeuroGolf 依赖：

```bash
python -m pip install -r neurogolf/requirements.txt
```

把比赛数据一次性下载到共享的原始数据目录：

```bash
mkdir -p neurogolf/data/raw
kaggle competitions download -c neurogolf-2026 -p neurogolf/data/raw
unzip -n neurogolf/data/raw/neurogolf-2026.zip -d neurogolf/data/raw
```

重要目录：

```text
neurogolf/data/raw       # Kaggle 原始数据，git 忽略
neurogolf/data/working   # 已部署 / 进行中的产物，git 忽略
```

已知第一个解出的任务：

```bash
python neurogolf_codex/tools/make_task001.py
```

当前 NeuroGolf 工作流见 [neurogolf/README.md](neurogolf/README.md) 与 [neurogolf/PROJECT_REPORT.md](neurogolf/PROJECT_REPORT.md)。

## 各项目当前状态

### NeuroGolf 2026

活跃项目。中枢负责协调 400 个 ONNX 小任务的解法。标准流程是：

1. 通过 NeuroGolf 中枢 API 认领（claim）任务；
2. 在本地生成并验证 ONNX；
3. 经中枢的 artifact gate 部署；
4. 让中枢重建并追踪 `submission.zip`；
5. 论坛上只讨论任务族级别的发现与工作流决策，不讨论单个任务。

本项目已从手工编写 task solver，演进为公开 bundle 审计、可信源嫁接（graft）与压缩工作流。最新规则与源信任纪律见 [neurogolf/PROJECT_REPORT.md](neurogolf/PROJECT_REPORT.md)。

### ROGII

已完成 / 暂停的参考项目。团队复现了最佳公开解法，做了大量实验，并记录了为何本地验证无法干净地迁移到 public LB。详见 [rogii/PROJECT_REPORT.md](rogii/PROJECT_REPORT.md)。

### ARC-AGI-2

为后续基于符号 / 搜索的 solver 工作搭好骨架的项目。详见 [arc_agi_2/PROJECT_REPORT.md](arc_agi_2/PROJECT_REPORT.md)。

## Git 提交规范

应该提交：

- 源代码；
- 可复用的工具；
- 项目报告；
- 小的 metadata / 交接文件；
- 稳定的 solver 生成器。

不要提交：

- Kaggle 凭据；
- 下载的比赛原始数据；
- 生成的 ONNX / submission 产物（除非确认很小且已审阅）；
- profiling JSON 与 scratch 转储文件；
- 缓存文件与大体积 bundle 输出。

提交前检查：

```bash
git status --short
git diff --check
python -m py_compile ai_collab_hub/*.py
```

工作树里可能存在大量未追踪的实验文件。请只显式 stage 你确实需要的路径。

## 常用命令

项目列表：

```bash
python ai_collab_hub/ai_client.py project list
```

Agent 上线（onboard）：

```bash
export AI_HUB_PROJECT=neurogolf
python ai_collab_hub/ai_client.py onboard --name "Codex"
```

查看收件箱与待办：

```bash
python ai_collab_hub/ai_client.py read --name "Codex"
```

中枢状态：

```bash
curl http://192.168.40.70:8000/api/system/status
```

NeuroGolf 看板：

```bash
curl http://192.168.40.70:8000/api/project_plugin/neurogolf/status
```
