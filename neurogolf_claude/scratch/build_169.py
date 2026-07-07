"""Build compact ONNX for task169.
Rule: 4-connected gray(5) components, recolor by size: size2->3, size3->2, size4->1.
Components are isolated (never touch even diagonally). All grids 10x10 inside 30x30 pad.

Algorithm (conv-only, exact):
 1. Collapse one-hot input -> color grid g30 [1,1,30,30] f32 via Conv weight [[0..9]].
 2. Crop to 10x10. gray mask m = (g==5).
 3. Anchor label A: idx = (row*10+col+1)*m ; propagate max over 4-neighborhood masked to m, 4 steps.
 4. size(p) = count of cells q in 7x7 window with A[q]==A[p] and A>0.
 5. color = (5 - size) on gray, else 0.  -> uint8 color grid gm [1,1,10,10].
 6. Pad gm to 30x30 (value 0). Equal(gm30, colors[1,10,1,1] uint8) -> output one-hot bool [1,10,30,30].
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

H = W = 10
G = 30

nodes = []
inits = []

def init(name, arr):
    inits.append(numpy_helper.from_array(arr, name))
    return name

# ---- 1. collapse input -> g30 [1,1,30,30] f32 ----
w_collapse = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
init("w_collapse", w_collapse)
nodes.append(helper.make_node("Conv", ["input", "w_collapse"], ["g30"],
                              kernel_shape=[1, 1]))

# ---- 2. crop to 10x10 ----
init("c_starts", np.array([0, 0], dtype=np.int64))
init("c_ends", np.array([H, W], dtype=np.int64))
init("c_axes", np.array([2, 3], dtype=np.int64))
nodes.append(helper.make_node("Slice", ["g30", "c_starts", "c_ends", "c_axes"], ["g10"]))

# gray mask m (float 1/0): Equal(g10, 5) -> bool -> cast float
init("five", np.array(5.0, dtype=np.float32).reshape(1, 1, 1, 1))
nodes.append(helper.make_node("Equal", ["g10", "five"], ["m_bool"]))
nodes.append(helper.make_node("Cast", ["m_bool"], ["m"], to=TensorProto.FLOAT))

# ---- 3. anchor label A ----
# idx grid (row*10+col+1) float
idx = (np.arange(H)[:, None] * W + np.arange(W)[None, :] + 1).astype(np.float32).reshape(1, 1, H, W)
init("idxgrid", idx)
nodes.append(helper.make_node("Mul", ["m", "idxgrid"], ["A0"]))

# max-dilation masked to m, 4 steps. Use MaxPool 3x3 then Mul by m.
# But MaxPool 3x3 includes diagonals; that's fine because anchor only spreads within comp
# only when masked by m afterwards, and diagonal cells of same comp don't exist (4-conn comp).
# However diagonal neighbor could belong to SAME comp? No: 4-connected comp can have cells that are
# diagonal to each other (e.g. L-shape). MaxPool 3x3 would let anchor jump diagonally across the
# corner of an L which IS same comp -> still same comp, fine. Cross-comp jump? comps separated by >=2
# chebyshev? min sep was 2 -> a 3x3 pool could touch a cell 1 away... but that cell is background(m=0)
# so after Mul by m it's killed. Distinct comps never within chebyshev 1 of each other's gray cells
# (min sep 2). So MaxPool3x3 + mask is safe. Need 4 steps for diameter.
prev = "A0"
for i in range(4):
    mp = f"Amp{i}"
    nodes.append(helper.make_node("MaxPool", [prev], [mp], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    am = f"A{i+1}"
    nodes.append(helper.make_node("Mul", [mp, "m"], [am]))
    prev = am
Aname = prev  # final anchor

# ---- 4. size = count of window cells with same A>0 ----
# For each shift (dy,dx) in [-3..3]^2: shifted A == A and A>0 -> contributes 1.
# Implement shift via Pad+Slice. We accumulate float counts.
# A>0 mask:
init("zero", np.array(0.0, dtype=np.float32).reshape(1, 1, 1, 1))
nodes.append(helper.make_node("Greater", [Aname, "zero"], ["Apos_bool"]))
nodes.append(helper.make_node("Cast", ["Apos_bool"], ["Apos"], to=TensorProto.FLOAT))

R = 3
acc = None
acc_idx = 0
for dy in range(-R, R + 1):
    for dx in range(-R, R + 1):
        # shifted A: pad then slice to realize shift by (dy,dx)
        # We want As[y,x] = A[y+dy, x+dx] (0 outside). Build via Pad with big margins then Slice.
        pad_top = max(dy, 0); pad_bottom = max(-dy, 0)
        pad_left = max(dx, 0); pad_right = max(-dx, 0)
        # onnx Pad pads format: [b0,c0,h0,w0, b1,c1,h1,w1]
        padded = f"Apad_{acc_idx}"
        nodes.append(helper.make_node("Pad", [Aname], [padded], mode="constant",
                                      pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
                                      value=0.0))
        # slice back to HxW: start at (pad_top - dy?) -- compute crop offset.
        # padded shape = H+pad_top+pad_bottom, W+pad_left+pad_right.
        # We want region corresponding to original index y in [0,H): As[y]=A[y+dy].
        # In padded coords, A[k] sits at k+pad_top. We want A[y+dy] -> at (y+dy)+pad_top.
        # For y in [0,H): index = y+dy+pad_top. start = dy+pad_top, end=start+H.
        s_y = dy + pad_top
        s_x = dx + pad_left
        sname = f"sst_{acc_idx}"; ename = f"sen_{acc_idx}"; axname = f"sax_{acc_idx}"
        init(sname, np.array([s_y, s_x], dtype=np.int64))
        init(ename, np.array([s_y + H, s_x + W], dtype=np.int64))
        init(axname, np.array([2, 3], dtype=np.int64))
        As = f"As_{acc_idx}"
        nodes.append(helper.make_node("Slice", [padded, sname, ename, axname], [As]))
        # eq = (As == A) & (A>0)
        eqb = f"eqb_{acc_idx}"
        nodes.append(helper.make_node("Equal", [As, Aname], [eqb]))
        eqf = f"eqf_{acc_idx}"
        nodes.append(helper.make_node("Cast", [eqb], [eqf], to=TensorProto.FLOAT))
        contrib = f"contrib_{acc_idx}"
        nodes.append(helper.make_node("Mul", [eqf, "Apos"], [contrib]))
        if acc is None:
            acc = contrib
        else:
            newacc = f"acc_{acc_idx}"
            nodes.append(helper.make_node("Add", [acc, contrib], [newacc]))
            acc = newacc
        acc_idx += 1
size = acc  # float per-gray-cell size, 0 on background

# ---- 5. color = (5 - size) on gray ----
init("fiveg", np.array(5.0, dtype=np.float32).reshape(1, 1, 1, 1))
nodes.append(helper.make_node("Sub", ["fiveg", size], ["colf_raw"]))
# colf_raw = 5 - size; on background size=0 -> 5, must be 0. multiply by m.
nodes.append(helper.make_node("Mul", ["colf_raw", "m"], ["colf"]))
# cast to uint8 color grid
nodes.append(helper.make_node("Cast", ["colf"], ["gm10"], to=TensorProto.UINT8))

# ---- 6. pad to 30x30 then Equal against colors ----
nodes.append(helper.make_node("Pad", ["gm10"], ["gm30"], mode="constant",
                              pads=[0, 0, 0, 0, 0, 0, G - H, G - W], value=0.0))
colors = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
init("colors", colors)
nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))

# ---- graph ----
inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, G, G])
out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, G, G])
graph = helper.make_graph(nodes, "task169", [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
model.ir_version = 10
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b11_task169.onnx")
print("saved. nodes:", len(nodes), "inits:", len(inits))
