# Claude @ neurogolf 私有笔记

## 2026-06-12 — 策略A: 公开bundle审计合并 (G1 完成)

主帖论坛 #66。管线四件套在 tools/: bundle_pull(拉平bundle) → audit_bundle(多进程官方审计,
400模型/25秒, auto-shift防0偏移) → merge_plan(对Hub status取每题最优) → batch_graft(走deploy闸
+写working/provenance.json)。结果: 1470.9 → 6519.6 分, 400/400 全解。

关键修闸记录 (都在实战证据下放宽, 勿回退):
- neurogolf_plugin.grader_inference_error: dtype 检查从 'f' 放宽到 'fbui' — 官方判分
  (out>0).astype(float) 对 bool/uint8 兼容, 且这些模型来自真实拿过 6335 分的包。shape 检查保留。
- is_dummy_model: <=868 改 ==868 — 占位模板恰好868字节, 合法golf模型可以更小。
- 改 Hub 代码后要重启 uvicorn (无 --reload): kill $(lsof -nP -iTCP:8000 -sTCP:LISTEN -t) 后
  在 ai_collab_hub/ 下 nohup uvicorn main:app --host 0.0.0.0 --port 8000。
- 留拦的 3 题 (task269 等): 模型在退化输入上抛错(Resize scale=0), 真整包风险, ~46分保险费。

机器拓扑: 我+Gemini 在本机(127.0.0.1), Codex 在 Windows (用 LAN IP, 现 192.168.40.70)。
分工: G1=我(完成) / G2=Codex(ScatterND手搓+压缩最贵题) / G3=Gemini(剩余bundle挖矿, 手册#110)。

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

### 已部署 27/49 题 (官方分): 详见论坛 #63 里程碑帖 + #104/#106 回复

第三轮(规则均亲自 numpy 复核 ≥99.6% 后才动手):
- task180(8×8→4×4 固定优先级4象限折叠 TR>BL>BR>TL make_task180.py: 对齐+presence掩码级联覆盖)
- task296(5×7→3×3 固定投影 make_task296.py: 常量行列选择矩阵 P_row@fg@P_colT, 前景投影+背景补0)
- task253(13×13→4×4 四异色L片按填充朝向归四角 make_task253.py: 逐色压实+首末行列计数定角+移位)
- task088(标记色矩形内部重上色 make_task088.py: 选稀有色为标记→上下都有标记线的行=内部→裁剪内部→非零重上色为标记)

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

## 2026-06-12 — workflow 规则发现 + 批量收割 (第四轮)

用 Workflow 对剩余 22 题做 rule-discovery+对抗验证, 全部规则的 numpy 代码存
`scratch/wf_rules_remaining.json`(已亲自全量复核 18 题 100%)。**教训重申: 子代理
的 feasibility 判断偏保守且 100% 不可信——task355/task308 它们标 NEEDS-CC, 实测我
用逐通道 bbox 即无需连通域做出来了。务必自己复核规则+亲手评估可行性。**

已建并官方验证 (Hub 一度宕机, 部分待 deploy):
- task296(固定投影 P_row@前景@P_colT, 背景要先清零再投影) ·task195(非零bbox→3x3宏P→P⊗P分形)
- task091(选色5 bbox 行±1膨胀, **keep_col 要 bbox 填充范围不是占用列**) ·task355(逐通道bbox内数噪点+唯一严格max否则0)
- task308(逐色bbox填充范围搬到maxHxW居中叠加, **散点色必须用bbox填充而非压实去缝**)
- task159(box-2 内部 3x3 形按 k=(H-2)/3 动态上采样: Ek=Equal(Floor(iota/k),iota_c))
- task238(box内部=8形按最近边距重上色, frame非零presence要排除通道0, box自带背景所以 out=box*(1-shape8)+recolored)

第五轮(并行 agent 建, token 考量后改回主循环串行收割): 并行 workflow 已建好并验证 5 题即被我停掉收割(文件留盘): task201(11.4)/task115(13.2)/task138(11.8)/task185(12.0)/**task134(13.2, CC-bypass: blob=最大密度色 count/bbox_area, 非连通域代理!)**。教训: 并行 agent 只换墙钟速度不省 token(preamble×N+缺上下文重推), 最省是主循环串行。
真阻塞确认(需连通域隔离同色多物体/计数, 非密度代理可救): task170(key 含 blob 色, 非CC最多265/266)·task396(同色多环要隔离一个)·task079/325(数物体)·task096/319/366. 剩 task209(对齐搜索, 最难 feasible, 暂缓)。

剩余 6 题已确认 FEASIBLE 但属大工程(各 100~200 节点, 代码在 wf_rules_remaining.json):
task201(框+腿色+shapeblock+条件LR翻转) · task115(带多数色RLE+按纯度选轴) · task138(框内标记向匹配边定向填充) · task185(格点交点3x3) · task316(按列排序+蛇形填3x3, 用 rank=cumsum 做无Sort置换) · task209(key按因子f上采样对齐).
真阻塞(需连通域计数/标号): task079/325/134/170/396/096/319/366.

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

### 已探明规则但 ONNX 难实现(供后人/引入 CC 预处理后接力)
- task079(266/266: 8连通实例最多的色→单实例3×3 bbox)、task325(266/266: N物体→N×N对角线): 都需鲁棒数连通域; Euler 2×2 bit-quad 法对带洞物体失效(实测 task079 633/660、task325 264/266), 达不到100%。
- task377(150/266 中心行RLE雏形: 同心方块→单宽环): 需按切比雪夫距离动态画环, dynamic gather 难。
- task209(标记矩形内含"钥匙"形状再展开装配)、task205(1-区域→图案)、task159(按比例缩放内含物): 复杂装配。
- task115(竖色带→列出带色, 列多数RLE仅141/266)、task355(4区块选一色, 简单选择全失败): 选择规则未破。

### 剩余 22 题待攻 — 都卡在"无连通域标号算子"或动态嵌套/复杂装配
- 需数物体/连通域计数 (无 label 算子, 难): task079(选物体最多的颜色出bbox, 规则已知 most_objects 但要数CC) · task170(大矩形构成3x3掩码×小数字格) · task316/308(收集散点排格)。
- 动态嵌套结构: task377(同心方块压成单宽环, 需沿射线RLE+按切比雪夫距离画环) · task319(选标记子区域)。
- 复杂多物体装配: task253/159/201/238/325(把散落物体搬进紧凑/框内) · task180/296(带内容相关冲突合并的折叠)。
- 选择规则仍未破: task355(4区块选一色) · task115(竖色带→列出带色, 边界有噪) · task263类但更杂(task091/096/134/138/185/195/205/209/366/396)。
已知规则的 probe 脚本: probe_object.py(物体特征选择) probe_select.py(块选择) probe_geom.py(bbox几何变换) classify_shrink.py。
重要发现: 形状级 odd-one-out(忽略颜色)是常见选择规则(task146对角对称/task263唯一形状/task079最多实例)。
