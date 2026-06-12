"""Golf kit: deterministic single-op pattern detectors + minimal ONNX builders + oracle.

策略: 在官方的 30x30 padding 空间里, 暴力检验每题能否被一个最小算子在全部
train+test+arc-gen 样例上 100% 解释。能解释 = 保证正确的最小模型(无毒, 因为是
我们自己的真算法解, 不是查表过拟合)。探测器只负责"提议", 最终一律由官方 oracle
(verify_network 跑全部样例) 验证通过 + 分数更高才算数。

得分 = max(1, 25 - ln(运行时内存字节 + 参数个数))。
排序优先级: Transpose(0参)=25 > 单 Gather(几十参)=21~22.7 > Slice+Pad。
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper

ROOT = "/Users/kanxiao/IdeaProjects/kaggletest"
RAW = os.path.join(ROOT, "neurogolf/data/raw")
UTILS_DIR = os.path.join(RAW, "neurogolf_utils")

_C, _H, _W = 10, 30, 30


# ---------- 样例加载 / 网格<->padded ----------
def load_examples(task_num: int) -> dict:
    with open(os.path.join(RAW, f"task{task_num:03d}.json")) as f:
        return json.load(f)


def all_pairs(task_num: int):
    """返回 [(in_grid, out_grid), ...] 覆盖 train+test+arc-gen, 跳过任一边 >30 的。"""
    ex = load_examples(task_num)
    out = []
    for split in ("train", "test", "arc-gen"):
        for p in ex.get(split, []):
            ig, og = p["input"], p["output"]
            if max(len(ig), len(ig[0]), len(og), len(og[0])) > 30:
                continue
            out.append((ig, og))
    return out


def pad30(grid) -> np.ndarray:
    a = np.zeros((_H, _W), dtype=np.int64)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            a[r, c] = v
    return a


# ---------- 探测器: 每个返回 None 或 (kind, params) ----------
def detect_identity(pairs):
    for ig, og in pairs:
        if pad30(ig).tolist() != pad30(og).tolist():
            return None
    return ("identity", {})


def detect_transpose(pairs):
    for ig, og in pairs:
        if not np.array_equal(pad30(ig).T, pad30(og)):
            return None
    return ("transpose", {})


def detect_color_perm(pairs):
    """存在固定颜色映射 P (P[0]=0) 使 out = P[in] 在全部样例成立。"""
    perm = {}
    for ig, og in pairs:
        ia, oa = pad30(ig), pad30(og)
        for a, b in zip(ia.flatten(), oa.flatten()):
            a, b = int(a), int(b)
            if a in perm:
                if perm[a] != b:
                    return None
            else:
                perm[a] = b
    # 必须是 0..9 上的合法映射, 且是双射(换色器需 Gather 索引完整)
    full = [perm.get(i, i) for i in range(10)]
    if perm.get(0, 0) != 0:
        return None
    if sorted(set(full)) != sorted(full):  # 非双射则 Gather 仍可表达(允许多对一)
        pass
    if all(full[i] == i for i in range(10)):
        return None  # 恒等, 交给 identity
    return ("color_perm", {"perm": full})


def _solve_axis_index(pairs, axis):
    """找固定 30 长索引 idx 使 out 沿 axis 是 in 的重排 (另一轴逐元素相等)。
    axis=0 行重排(out[i,:]=in[idx[i],:]); axis=1 列重排(out[:,j]=in[:,idx[j]])。"""
    cand = [set(range(30)) for _ in range(30)]
    for ig, og in pairs:
        ia, oa = pad30(ig), pad30(og)
        if axis == 0:
            for i in range(30):
                ok = {k for k in cand[i] if np.array_equal(oa[i, :], ia[k, :])}
                cand[i] = ok
        else:
            for j in range(30):
                ok = {k for k in cand[j] if np.array_equal(oa[:, j], ia[:, k])}
                cand[j] = ok
        if any(len(s) == 0 for s in cand):
            return None
    # 取每位最小可行索引; 偏好 idx[i]=i 以减小后续二轴探测干扰
    idx = [min(cand[i], key=lambda k: (k != i, k)) for i in range(30)]
    return idx


def detect_row_gather(pairs):
    idx = _solve_axis_index(pairs, 0)
    if idx is None or idx == list(range(30)):
        return None
    return ("row_gather", {"idx": idx})


def detect_col_gather(pairs):
    idx = _solve_axis_index(pairs, 1)
    if idx is None or idx == list(range(30)):
        return None
    return ("col_gather", {"idx": idx})


def detect_rowcol_gather(pairs):
    """二轴联合 out[i,j]=in[ri[i],cj[j]] (覆盖 180 旋转/翻转拼接等固定重排)。"""
    ri = _solve_axis_index(pairs, 0)
    cj = _solve_axis_index(pairs, 1)
    # 单轴已能解时不在这里(交给单轴更省). 仅当两轴都非恒等才有意义
    if ri is None:
        ri = list(range(30))
    if cj is None:
        cj = list(range(30))
    if ri == list(range(30)) and cj == list(range(30)):
        return None
    # 验证联合映射
    ri_a, cj_a = np.array(ri), np.array(cj)
    for ig, og in pairs:
        ia, oa = pad30(ig), pad30(og)
        if not np.array_equal(ia[np.ix_(ri_a, cj_a)], oa):
            return None
    if ri == list(range(30)) or cj == list(range(30)):
        return None  # 退化成单轴, 让单轴探测器处理
    return ("rowcol_gather", {"ri": ri, "cj": cj})


def _find_color_perm(pairs_arr):
    """pairs_arr: [(in30,out30)] 求固定换色映射, 不一致返回 None。"""
    perm = {}
    for ia, oa in pairs_arr:
        for a, b in zip(ia.flatten(), oa.flatten()):
            a, b = int(a), int(b)
            if a in perm and perm[a] != b:
                return None
            perm[a] = b
    if perm.get(0, 0) != 0:
        return None
    return [perm.get(i, i) for i in range(10)]


def detect_crop(pairs):
    """固定窗口裁剪: out == in30[r0:r0+oh, c0:c0+ow], (r0,c0,oh,ow) 全样例一致。单 Slice。"""
    r0 = c0 = oh = ow = None
    for ig, og in pairs:
        ia = pad30(ig)
        H, W = len(og), len(og[0])
        if oh is None:
            oh, ow = H, W
            # 在 input 里找 output 块的位置(用第一对定位, 后续验证)
            tgt = pad30(og)[:H, :W]
            found = None
            for rr in range(31 - H):
                for cc in range(31 - W):
                    if np.array_equal(ia[rr:rr + H, cc:cc + W], tgt):
                        found = (rr, cc)
                        break
                if found:
                    break
            if not found:
                return None
            r0, c0 = found
        if (H, W) != (oh, ow):
            return None  # 输出尺寸必须固定才能用固定 Slice
        if not np.array_equal(ia[r0:r0 + oh, c0:c0 + ow], pad30(og)[:oh, :ow]):
            return None
    if (r0, c0, oh, ow) == (0, 0, 30, 30):
        return None
    return ("crop", {"r0": r0, "c0": c0, "oh": oh, "ow": ow})


def detect_transpose_color(pairs):
    """转置后换色: out == colorperm(in30.T)。"""
    arr = [(pad30(i).T, pad30(o)) for i, o in pairs]
    perm = _find_color_perm(arr)
    if perm is None:
        return None
    # 排除纯转置(交给单算子)与纯换色
    if perm == list(range(10)):
        return None
    return ("transpose_color", {"perm": perm})


def detect_gather_color(pairs):
    """行列重排后换色: out == colorperm(in30[ri][:,cj])。先定位 ri/cj(按颜色集合无关的结构)。
    简化: 仅当存在 ri,cj 使各样例几何重排后再套固定换色一致。用第一对的 0/非0 模式定位较脆,
    这里退化为'纯换色已 detect_color_perm 覆盖', 仅补'行/列重排+换色'。"""
    for axis_name, idx_fn in [("row", lambda: _solve_axis_index(pairs, 0)),
                              ("col", lambda: _solve_axis_index(pairs, 1))]:
        # 几何重排索引应与颜色无关; 用'是否背景0'的二值图求索引最稳
        bin_pairs = []
        ok = True
        for ig, og in pairs:
            bin_pairs.append(((pad30(ig) != 0).astype(np.int64), (pad30(og) != 0).astype(np.int64)))
        idx = _solve_axis_index_arr(bin_pairs, 0 if axis_name == "row" else 1)
        if idx is None or idx == list(range(30)):
            continue
        ax = np.array(idx)
        arr = []
        for ig, og in pairs:
            ia = pad30(ig)
            reordered = ia[ax, :] if axis_name == "row" else ia[:, ax]
            arr.append((reordered, pad30(og)))
        perm = _find_color_perm(arr)
        if perm is not None and perm != list(range(10)):
            return ("gather_color", {"axis": 2 if axis_name == "row" else 3,
                                     "idx": idx, "perm": perm})
    return None


def _solve_axis_index_arr(pairs_arr, axis):
    cand = [set(range(30)) for _ in range(30)]
    for ia, oa in pairs_arr:
        for i in range(30):
            if axis == 0:
                ok = {k for k in cand[i] if np.array_equal(oa[i, :], ia[k, :])}
            else:
                ok = {k for k in cand[i] if np.array_equal(oa[:, i], ia[:, k])}
            cand[i] = ok
        if any(len(s) == 0 for s in cand):
            return None
    return [min(cand[i], key=lambda k: (k != i, k)) for i in range(30)]


DETECTORS = [detect_transpose, detect_color_perm, detect_row_gather,
             detect_col_gather, detect_rowcol_gather, detect_crop,
             detect_transpose_color, detect_gather_color, detect_identity]


# ---------- ONNX 构造 (输入/输出 [1,10,30,30] float) ----------
def _model(nodes, inits, opset=13):
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, _C, _H, _W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, _C, _H, _W])
    g = helper.make_graph(nodes, "golf", [x], [y], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    return m


def build(kind, params):
    if kind == "identity":
        return _model([helper.make_node("Identity", ["input"], ["output"])], [])
    if kind == "transpose":
        return _model([helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])], [])
    if kind == "color_perm":
        idx = helper.make_tensor("ci", TensorProto.INT64, [10], params["perm"])
        return _model([helper.make_node("Gather", ["input", "ci"], ["output"], axis=1)], [idx])
    if kind == "row_gather":
        idx = helper.make_tensor("ri", TensorProto.INT64, [30], params["idx"])
        return _model([helper.make_node("Gather", ["input", "ri"], ["output"], axis=2)], [idx])
    if kind == "col_gather":
        idx = helper.make_tensor("ci", TensorProto.INT64, [30], params["idx"])
        return _model([helper.make_node("Gather", ["input", "ci"], ["output"], axis=3)], [idx])
    if kind == "rowcol_gather":
        ri = helper.make_tensor("ri", TensorProto.INT64, [30], params["ri"])
        cj = helper.make_tensor("cj", TensorProto.INT64, [30], params["cj"])
        return _model([
            helper.make_node("Gather", ["input", "ri"], ["t"], axis=2),
            helper.make_node("Gather", ["t", "cj"], ["output"], axis=3),
        ], [ri, cj])
    if kind == "crop":
        r0, c0, oh, ow = params["r0"], params["c0"], params["oh"], params["ow"]
        starts = helper.make_tensor("st", TensorProto.INT64, [2], [r0, c0])
        ends = helper.make_tensor("en", TensorProto.INT64, [2], [r0 + oh, c0 + ow])
        axes = helper.make_tensor("ax", TensorProto.INT64, [2], [2, 3])
        return _model([helper.make_node("Slice", ["input", "st", "en", "ax"], ["output"])],
                      [starts, ends, axes])
    if kind == "transpose_color":
        ci = helper.make_tensor("ci", TensorProto.INT64, [10], params["perm"])
        return _model([
            helper.make_node("Transpose", ["input"], ["t"], perm=[0, 1, 3, 2]),
            helper.make_node("Gather", ["t", "ci"], ["output"], axis=1),
        ], [ci])
    if kind == "gather_color":
        gi = helper.make_tensor("gi", TensorProto.INT64, [30], params["idx"])
        ci = helper.make_tensor("ci", TensorProto.INT64, [10], params["perm"])
        return _model([
            helper.make_node("Gather", ["input", "gi"], ["t"], axis=params["axis"]),
            helper.make_node("Gather", ["t", "ci"], ["output"], axis=1),
        ], [gi, ci])
    raise ValueError(kind)


# ---------- 官方 oracle ----------
def oracle(onnx_path: str, task_num: int):
    """跑官方 verify_network, 返回 (passed, score, raw)。在临时 cwd 跑(它会写 profiling)。"""
    import re
    cwd = os.getcwd()
    abs_onnx = os.path.abspath(onnx_path)  # 必须在 chdir 之前解析, 否则相对路径会指错
    tmp = tempfile.mkdtemp()
    if UTILS_DIR not in sys.path:
        sys.path.insert(0, UTILS_DIR)
    try:
        os.chdir(tmp)
        import neurogolf_utils as nu
        buf = io.StringIO()
        # verify_network 末尾的 display(FileLink) 在 headless 会抛异常, 但 IS READY/分数
        # 此时已打进 buf, 所以异常照样解析 buf, 不能直接判 0。
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                nu.verify_network(abs_onnx, task_num)
            except Exception:  # noqa: BLE001
                pass
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
    passed = "IS READY" in out
    m = re.search(r"yielding\s+([\d.]+)\s+points", out)
    score = float(m.group(1)) if m else (1.0 if passed else 0.0)
    return (passed, score, out[-400:])


def sweep_one(task_num: int):
    """对一题跑全部探测器, 第一个能解释全样例的 -> 构造+oracle验证 -> 返回结果。"""
    pairs = all_pairs(task_num)
    if not pairs:
        return {"task": task_num, "fit": None, "reason": "no pairs"}
    for det in DETECTORS:
        hit = det(pairs)
        if not hit:
            continue
        kind, params = hit
        try:
            model = build(kind, params)
        except Exception as e:  # noqa: BLE001
            return {"task": task_num, "fit": kind, "built": False, "err": str(e)}
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        onnx.save(model, tmp.name)
        passed, score, raw = oracle(tmp.name, task_num)
        return {"task": task_num, "fit": kind, "built": True, "onnx": tmp.name,
                "passed": passed, "score": round(score, 3), "raw": raw if not passed else ""}
    return {"task": task_num, "fit": None, "reason": "no detector matched"}


if __name__ == "__main__":
    for tn in [int(a) for a in sys.argv[1:]]:
        print(sweep_one(tn))
