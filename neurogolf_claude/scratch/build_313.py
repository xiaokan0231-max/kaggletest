"""Compact ONNX for task313 (v2 - minimal ops).
Rule: input = periodic anti-diagonal stripe texture (row-period 2, col-period pc in
{2,3}) + junk L-border. Output = texture extended over the grid, shifted LEFT by 1
column. Junk = cell equal to an orthogonal neighbour. All interior work uint8 on a
21x21 crop. Final Equal -> bool one-hot = graph OUTPUT (excluded from memory)."""
import numpy as np
import onnx
from onnx import helper, TensorProto

S = 21
U8, BOOL, F32, I64, I32 = (TensorProto.UINT8, TensorProto.BOOL, TensorProto.FLOAT,
                           TensorProto.INT64, TensorProto.INT32)
SENT = 255
GS = [1, 1, S, S]
nodes, inits, vinfo = [], [], []


def C(name, arr, dtype):
    a = np.asarray(arr)
    inits.append(helper.make_tensor(name, dtype, a.shape, a.flatten().tolist()))


def VI(name, shape, etype):
    vinfo.append(helper.make_tensor_value_info(name, etype, shape))


def N(op, ins, outs, shape=None, etype=U8, **kw):
    nodes.append(helper.make_node(op, ins, outs, **kw))
    if shape is not None:
        VI(outs[0], shape, etype)


# 1. one-hot -> color grid (single irreducible f32 op), crop to SxS, cast uint8
C("Wcolor", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
N("Conv", ["input", "Wcolor"], ["g30f"], [1, 1, 30, 30], F32)
C("cs", [0, 0, 0, 0], I64)
C("ce", [1, 1, S, S], I64)
N("Slice", ["g30f", "cs", "ce"], ["gSf"], GS, F32)
N("Cast", ["gSf"], ["g"], GS, U8, to=U8)

# ingrid mask
C("Wones", np.ones((1, 10, 1, 1), dtype=np.float32), F32)
N("Conv", ["input", "Wones"], ["sum30"], [1, 1, 30, 30], F32)
N("Slice", ["sum30", "cs", "ce"], ["sumS"], GS, F32)
C("half", [0.5], F32)
N("Greater", ["sumS", "half"], ["ingrid"], GS, BOOL)
N("Not", ["ingrid"], ["outgrid"], GS, BOOL)

C("sentS", [SENT], U8)
C("sentG", np.full(GS, SENT, np.uint8), U8)


def shift(src, dr, dc, out):
    """out[r,c] = src[r-dr, c-dc]; vacated -> 255 (Slice + Pad)."""
    rs, re_ = max(0, -dr), S - max(0, dr)
    csv, cev = max(0, -dc), S - max(0, dc)
    C(out + "_s", [0, 0, rs, csv], I64)
    C(out + "_e", [1, 1, re_, cev], I64)
    N("Slice", [src, out + "_s", out + "_e"], [out + "_sl"], [1, 1, re_ - rs, cev - csv], U8)
    C(out + "_p", [0, 0, max(0, dr), max(0, dc), 0, 0, max(0, -dr), max(0, -dc)], I64)
    N("Pad", [out + "_sl", out + "_p", "sentS"], [out], GS, U8)


# 2. junk mask: cell equals its BELOW neighbour OR its LEFT neighbour (2 dirs suffice).
shift("g", -1, 0, "g_dn")   # g_dn[r,c] = g[r+1,c]  (cell below)
shift("g", 0, 1, "g_lf")    # g_lf[r,c] = g[r,c-1]  (cell to the left)
N("Equal", ["g", "g_dn"], ["e1"], GS, BOOL)
N("Equal", ["g", "g_lf"], ["e2"], GS, BOOL)
N("Or", ["e1", "e2"], ["Jraw"], GS, BOOL)
N("And", ["Jraw", "ingrid"], ["J"], GS, BOOL)
N("Or", ["J", "outgrid"], ["unk0"], GS, BOOL)
N("Where", ["unk0", "sentG", "g"], ["T0"], GS, U8)


def fillstep(src, dr, dc, out):
    """out = where(src==255, src_shifted_by(dr,dc), src). 4 tensors."""
    shift(src, dr, dc, out + "_sh")
    N("Equal", [src, "sentS"], [out + "_u"], GS, BOOL)
    N("Where", [out + "_u", out + "_sh", src], [out], GS, U8)


# 3. vertical period-2 fill (bottom strip)
cur = "T0"
for i, k in enumerate([2, 4, 8]):
    fillstep(cur, k, 0, f"V{i}")
    cur = f"V{i}"
Tv = cur


def hchain(prefix, pc):
    c = Tv
    for j, m in enumerate([1, 2, 4]):
        fillstep(c, 0, m * pc, f"{prefix}{j}")
        c = f"{prefix}{j}"
    return c


A = hchain("A", 2)
B = hchain("B", 3)

# 5. detect pc with a single-cell test: is_pc2 = (g[0,0] == g[0,2]).
#    Top-left 1x3 is always clean; equal => period 2, differ => period 3.
C("c00s", [0, 0, 0, 0], I64)
C("c00e", [1, 1, 1, 1], I64)
N("Slice", ["g", "c00s", "c00e"], ["g00"], [1, 1, 1, 1], U8)
C("c02s", [0, 0, 0, 2], I64)
C("c02e", [1, 1, 1, 3], I64)
N("Slice", ["g", "c02s", "c02e"], ["g02"], [1, 1, 1, 1], U8)
N("Equal", ["g00", "g02"], ["is_pc2"], [1, 1, 1, 1], BOOL)

# 6. blend, 7. left-shift, 8. mask out-of-grid, 9. one-hot output
N("Where", ["is_pc2", A, B], ["Tfull"], GS, U8)
shift("Tfull", 0, -1, "Lsh")
N("Where", ["ingrid", "Lsh", "sentG"], ["gm"], GS, U8)
C("padG", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S], I64)
N("Pad", ["gm", "padG", "sentS"], ["gm30"], [1, 1, 30, 30], U8)
C("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))

x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
graph = helper.make_graph(nodes, "task313", [x], [y], inits, value_info=vinfo)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b19_task313.onnx")
print("saved. nodes:", len(nodes))
