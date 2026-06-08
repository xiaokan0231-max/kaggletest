# Codex 交接任务书：在 RTX 4060 Windows 上搭建 Kaggle 方案的「完整离线复现环境」

> 你（Codex）将在一台 **RTX 4060 的 Windows 笔记本**上工作。目标是把一个 Kaggle 竞赛的公开 Notebook 方案**完整地在本地跑通**，产出 `submission.csv`，并让若干关键中间数值与云端参考值**逐一对上**。这样我们之后就能在本地端到端验证算法改进，而不必每次都赌 Kaggle 几小时的排队评分。
>
> **重要：你不需要、也不要向 Kaggle 提交任何东西。** 你只负责"本地能跑通 + 数值对得上"。提交由主控（Mac 上的另一个 agent）统一管理。

---

## 1. 背景（30 秒）

- 竞赛：`rogii-wellbore-geology-prediction`（ROGII 井眼地质预测，Featured，$50k，截止 2026-08-05）。
- 任务：水平井在「预测起点 PS」之后的 `TVT`（真垂直厚度）值，用 PS 之前已知的 `TVT_input` + `GR`（伽马）+ 典型井曲线来预测。评分是预测段的 RMSE。
- 这是 **Code competition**：提交的是 Notebook，Kaggle 重跑它生成 `submission.csv`。
- 我们复现的方案（公榜约 **7.904**）结构：
  `最终 = 0.3 × GBDT栈(LightGBM+CatBoost→Ridge) + 0.7 × 启发式(粒子滤波+Beam+物理模型)`
- **关键利好**：GBDT 模型是**预训练好的产物**（在 artifacts 数据集里），Notebook 是**加载**而非重训——所以本地复现**不强依赖 GPU 训练**，CPU 也能跑通（只是某些 cell 默认写了 `device="gpu"`，需要时改成 CPU，见 §6）。

---

## 2. 最终交付物（验收清单）

1. ✅ 一个**可一键运行**的本地脚本/Notebook，跑完产出 `submission.csv`。
2. ✅ `submission.csv`：**14151 行**，列为 `['id','tvt']`，无 NaN，`tvt` 取值范围约 `[11587.8, 12239.4]`。
3. ✅ 运行日志里这些**参考数值要对上**（这是正确性铁证，见 §7）：
   - `Loaded lightgbm-1 ... overall RMSE: 10.7668`，lightgbm-2 `10.4852`，lightgbm-3 `10.4733`
   - `Loaded catboost-1 ... 10.5750`，catboost-2 `10.5550`
   - `Baseline pp score: 10.37225`
   - `Selected pp score: 10.37058`
4. ✅ 一份 `LOCAL_SETUP.md`：写清你在 Windows 上**实际执行的每一步命令**（conda 环境、装包、数据放哪、改了哪些路径/cell），让别人能照着复现。
5. ✅ 把 `submission.csv` + 完整运行日志 + `LOCAL_SETUP.md` 放在仓库里交回。

---

## 3. 机器与前置

- 机器：RTX 4060 Windows 笔记本。建议用 **Miniconda/conda** 建独立环境。
- **Python 3.11**（不要用 3.13；numba/koolbox 在 3.13 上易出问题——这是 Mac 端踩过的坑）。
- 需要 Kaggle API（下载数据用）。如果这台 Windows 还没配：
  - 装 `pip install kaggle`
  - 把 Kaggle token 放到 `C:\Users\<你>\.kaggle\kaggle.json`（旧格式）或对应 `access_token` 文件。
  - 注意：**新格式 `KGAT_...` 的 token 要放 `~/.kaggle/access_token` 纯文本文件，不是 kaggle.json**（Mac 端踩过：放 kaggle.json 会报 "Authentication required"）。token 向用户索取。

---

## 4. 需要拉取的素材（全部用 Kaggle CLI）

```bash
# 1) 竞赛数据（含 train/ test/ sample_submission.csv）
kaggle competitions download -c rogii-wellbore-geology-prediction -p ./data
#   解压后应有: train/  test/  sample_submission.csv
#   train/ 约 773 口井, 每口 <id>__horizontal_well.csv + <id>__typewell.csv
#   test/  只有 3 口井（注意：这 3 口其实也在 train 里，是带泄漏的样例——别纠结，照跑）

# 2) koolbox 离线包（Notebook 依赖的自定义训练库 Trainer）
kaggle datasets download -d phongnguyn23021656/koolbox-offline -p ./koolbox-offline
#   里面有 koolbox-0.1.3-py3-none-any.whl，pip 装它即可

# 3) 预训练产物（GBDT 模型 + 预计算特征 train.csv）——这是本地能跳过重训的关键
kaggle datasets download -d ravaghi/wellbore-geology-prediction-artifacts -p ./artifacts
#   解压后应有 data/train.csv 和 models/lightgbm-* models/catboost-* 等 .pkl

# 4) 要复现的 Notebook（公开原版，未改动，复现目标=公榜7.904）
kaggle kernels pull lightningv08/lb-7-776-rogii-ridge-sp -p ./nb -m
#   得到 .ipynb；用 jupyter nbconvert --to script 转成 .py 更好调试
```

> 备注：主控这边的 fork 叫 `kanxiao0230/rogii-ridge-sp-fork`，v1 与原版等价（v2 加了 XGBoost 但效果变差，**别用 v2**）。直接用 `lightningv08/lb-7-776-rogii-ridge-sp` 原版即可。

---

## 5. 路径重映射（最容易卡的地方）

Notebook 里写死了 Kaggle 云端路径（见其 `class CFG`）：

```python
class CFG:
    dataset_path   = Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    artifacts_path = Path("/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts")
```

**两种解法，任选其一：**

- **(A) 改 CFG**（推荐，最简单）：把这两行指向你本地解压目录，例如
  ```python
  dataset_path   = Path(r"./data")          # 里面有 train/ test/ sample_submission.csv
  artifacts_path = Path(r"./artifacts")     # 里面有 data/train.csv, models/...
  ```
- **(B) 造目录树**：在本地建 `kaggle/input/...` 同名目录并把数据放进去（或用目录联接 `mklink /J`），不改代码。

还要处理 **koolbox 安装 cell**（Notebook 第 2 个 code cell 会尝试从 `/kaggle/input/koolbox-offline` 找 wheel）：本地直接 `pip install ./koolbox-offline/koolbox-0.1.3-py3-none-any.whl --no-deps`，并把那个 cell 改成无操作或让它找到你的本地 wheel 路径。

---

## 6. 依赖与 GPU/CPU

conda 环境大致需要：

```bash
conda create -n rogii python=3.11 -y && conda activate rogii
pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn numba
pip install lightgbm catboost xgboost
pip install ./koolbox-offline/koolbox-0.1.3-py3-none-any.whl --no-deps
pip install kaggle
```

**GPU 注意**：Notebook 里 GBDT 的 cell 默认写了 `device_type="gpu"` / `task_type="GPU"` / `device="gpu"`。但**这些模型是从 artifacts 加载、不重训**，所以理论上不会触发 GPU 训练。**只要 artifacts 里的 .pkl 能被加载，GPU 有没有都无所谓。**
- 如果某处仍尝试用 GPU 且报错（如 CatBoost/LightGBM 没装 GPU 版），把对应参数改成 CPU：`device_type="cpu"` / `task_type="CPU"` / 删掉 `device="gpu"`。
- 4060 装了 CUDA 的话 GPU 版也行，但**不是硬要求**——优先保证"能跑通+数值对上"。

---

## 7. 验收：数值必须对上（这是这份任务的灵魂）

跑通后，日志里应出现（来自云端 v1 真实运行，作为你的标准答案）：

```
Loaded lightgbm-1 with overall RMSE: 10.7668
Loaded lightgbm-2 with overall RMSE: 10.4852
Loaded lightgbm-3 with overall RMSE: 10.4733
Loaded catboost-1 with overall RMSE: 10.5750
Loaded catboost-2 with overall RMSE: 10.5550
...
Baseline pp score: 10.37225 params={'alpha': 1.0, 'tau': 85, 'w_pf': 0.09}
Selected pp score: 10.37058 params={'alpha': 1.02, 'tau': 105, 'w_pf': 0.09}
```

- 若 GBDT 的 RMSE 对上 → 说明 artifacts 加载正确、特征 `train.csv` 对齐。
- 若 `Baseline pp score = 10.37225` 对上 → 说明整条 GBDT+Ridge+后处理链路完全复现。
- `submission.csv` 14151 行 → 启发式分支（PF/Beam/物理模型）也跑通了。

**只要这几个数对上，就算复现成功。** 有微小末位差异（±0.001）可接受并记录；若差异大，请排查路径/版本/CPU-GPU 差异并在 `LOCAL_SETUP.md` 里写明。

---

## 8. 边界与注意

- ❌ **不要向 Kaggle 提交**任何东西（提交额度每天只有 5 次，由主控统一管理）。
- ❌ 不要去改算法逻辑/调参（那是主控在 Mac 上做的事）。你的任务**纯粹是"忠实复现 + 跑通 + 数值对上"**。
- ✅ 允许改的只有：路径（CFG）、koolbox 安装 cell、GPU→CPU 的 device 参数、把 ipynb 转 py 方便运行。这些改动都要记进 `LOCAL_SETUP.md`。
- ⚠️ 如果 `kaggle kernels pull` 因权限/版本问题失败，可改用网页端下载该 notebook 的 ipynb，或向用户索取。
- ⚠️ numba 首次运行会 JIT 编译，较慢属正常；整体跑一遍预计几分钟到十几分钟。

---

## 9.（可选加分）为后续迭代铺路

如果主线完成且有余力，做一个**最小评估钩子**：把整条 pipeline 包成一个函数 `run_pipeline() -> (submission_df, scores_dict)`，其中 `scores_dict` 至少含 `ridge_oof_rmse`、`baseline_pp_score`。这样主控之后改进算法时，能一行调用就拿到 OOF 分数对比。**不强求，主线优先。**

---

### 交回清单
1. `LOCAL_SETUP.md`（每一步命令 + 改了什么）
2. 完整运行日志（含 §7 的数值）
3. 生成的 `submission.csv`
4. 转好的可运行 `.py`（如果你做了 ipynb→py）

完成后告诉主控：「复现成功，数值对上：Baseline pp score = ___，submission ___ 行」。
