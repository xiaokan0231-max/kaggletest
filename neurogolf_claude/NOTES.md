# Claude @ neurogolf 私有笔记

> 只存"指针与偏好": 我的实验在哪个文件、下一步打算、个人观点。
> 不要复制论坛结论或项目正典内容 (防漂移); 共享产物放项目共享层。

## 2026-06-11 — task005 收割 + submission.zip 修复

> 注意: 本日早些时候我误用 "Codex" 身份在论坛操作过一轮（投票/实验#18/回复#92/
> 帖子#37/结案#22#26），已在 #37 帖内公开澄清并恢复 Codex 状态。教训:
> 论坛身份恒为 Claude，与用户口头给的目录名无关。

### Solved: task005 (11.616 points)

- Generator: `tools/make_task005.py`; numpy reference: `tools/ref_task005.py`
- 模板选色必须用 3x3 窗口密度而非总像素 argmax（arc-gen 有 2 例打平；
  窗口分数 top1-top2 最小差距 2）。
- Official: ARC-AGI 4/4, ARC-GEN 262/262, 648148 bytes + 1308 params。
- 压缩想法暂存: 5 个 Conv 级联可换单个 [8,1,41,41] ray-conv（估 ~460KB）。
  论坛 #26（已结案）。

### Tooling: submission.zip rebuild

`tools/rebuild_submission.py` — 把 data/working 的 400 个模型以权重内嵌
单文件重新序列化后打包，并对 zip 内逐个做独立加载校验。每次收割落盘后必须跑。
防两个隐患: zip 过期 dummy、`.onnx.data` 旁挂文件不入包（论坛 #37）。
注意: v2 用 Hub `/deploy` API 部署 (tools/hub_deploy.py), 会自动重建 zip, 不必手跑。

## 2026-06-11 (下午) — SHRINK/SAME 批量收割 16 题

冷启动指针: `export AI_HUB_PROJECT=neurogolf`; 任务筛 `neurogolf/data/working/task_index.csv`
里 rule_family=SHRINK & color_delta=SAME (共49)。Hub 状态/历史: `/api/project_plugin/neurogolf/{status,history}`。

### 通用裁剪-压实后端 `tools/cropkit.py`  ← 核心, 复用率最高
- I/O 都是 [1,10,30,30] one-hot; 解码器裁掉尾部全 0 → SHRINK 答案放左上即可。
- `emit_compaction(n, keep_row, keep_col, content, out)`: Sr@content@T 把选中行列搬左上。
  每题只需算 keep_row[1,1,30,1]/keep_col[1,1,1,30] 两个掩码。
- 配套: `colorkit.py`(选色 rarest/common→Conv 出色掩码), `make_crop.py`(first/last/frac/nz带/半选 DSL)。

### 已部署 23/49 题 (官方分): 详见论坛 #63 里程碑帖 + #104 回复
task300/310(bbox选色 make_bbox.py) · task067/135/326(角/分数裁剪 make_crop.py) ·
task039(nzbbox左上1/4) · task188(沿长轴去重 make_halvelong.py) ·
task065/207(异类象限 make_oddquad.py) · task178(条纹游程 make_task178.py) ·
task014/021/022/029/031/036(预建补部署) ·
task146(三叠块选非对角对称 make_task146.py: 整图Transpose就地转置TL块) ·
task109(中心十字, 左上象限重上色为分隔色+4折反射 make_task109.py) ·
task384(非零bbox整体2x上采样 make_task384.py: E@X@E^T, E[i,i//2]=1) ·
task057(非零bbox水平2x平铺 make_task057.py: X+右移) ·
task244(分隔线NxN块取色+左右镜像 make_task244.py: 检测首条满色行得步长+采样矩阵) ·
task174(选左右对称的单色物体出bbox make_task174.py: 9色循环各自压实+水平翻转比对) ·
task263(条纹中选唯一异形块 make_task263.py: 条件转置归一为竖条+5块形状两两比对取唯一)。

### 新增可复用 ONNX 技巧 (都靠结构化矩阵 MatMul)
- 反射/镜像: 反转矩阵 Equal(iota_r+iota_c, k) 右乘镜像列(绕 k/2), 左乘镜像行。
- 上采样2x: 固定 E[i,i//2]=1, up = E@X@E^T。 平铺: X + X右移w列。
- 块采样/降采样: 采样矩阵 Equal(iota_c, iota_r*stride), us = S@X@S^T。
- 就地转置每个TL对齐块: 整图 Transpose perm=[0,1,3,2] (块锚定原点且方形时成立)。
- 方向归一: 条件转置 combo = isv*X + ish*Xt, 末尾再转回。
- 逐色处理: 9次循环, 每色 Conv 取通道→压实→翻转/比对, 汇成 one-hot 选择器。

### 关键坑
- ReduceSum 的 axes 在 opset17 是**输入**不是属性 (task014 老脚本里就是这么写的); ReduceMax 仍属性。
- MatMul 移位 ([1,10,30,30]@[30,30]) 会把内存撑到 ~95万字节, 压到 ~11 分; 纯掩码裁剪保持 ~14 分。能不移位就别移位。
- 验证: `tools/verify_local.py taskNNN`; 必须看全量 ARC-GEN 262 零失误。

### 剩余 26 题待攻 — 都卡在"无连通域标号算子"或动态嵌套/复杂装配
- 需数物体/连通域计数 (无 label 算子, 难): task079(选物体最多的颜色出bbox, 规则已知 most_objects 但要数CC) · task170(大矩形构成3x3掩码×小数字格) · task316/308(收集散点排格)。
- 动态嵌套结构: task377(同心方块压成单宽环, 需沿射线RLE+按切比雪夫距离画环) · task319(选标记子区域)。
- 复杂多物体装配: task253/159/201/238/325(把散落物体搬进紧凑/框内) · task180/296(带内容相关冲突合并的折叠)。
- 选择规则仍未破: task355(4区块选一色) · task115(竖色带→列出带色, 边界有噪) · task263类但更杂(task091/096/134/138/185/195/205/209/366/396)。
已知规则的 probe 脚本: probe_object.py(物体特征选择) probe_select.py(块选择) probe_geom.py(bbox几何变换) classify_shrink.py。
重要发现: 形状级 odd-one-out(忽略颜色)是常见选择规则(task146对角对称/task263唯一形状/task079最多实例)。
