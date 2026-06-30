"""Compact ONNX (opset 10) for task090 — float16 run-length doubling.
 All arithmetic tensors float16 ([1,10,1,30]=600B). Equal replaced by Greater(x, k-0.5).
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

bands = [(r0, r1, r1 - r0 + 1) for r0 in range(5) for r1 in range(r0 + 1, 5)]
B = len(bands)
SHIFTS = [1, 2, 4, 8]
F16 = TensorProto.FLOAT16

nodes, inits = [], []
def C(name, arr):
    inits.append(numpy_helper.from_array(arr, name)); return name
def Cf16(name, val):
    return C(name, np.array(val, np.float16))

# Z = input[:,0:1,0:5,:] -> [1,1,5,30] f32, then Cast to f16
C("z_s", np.array([0, 0, 0, 0], np.int64)); C("z_e", np.array([1, 1, 5, 30], np.int64)); C("z_a", np.array([0, 1, 2, 3], np.int64))
nodes.append(helper.make_node("Slice", ["input", "z_s", "z_e", "z_a"], ["Zf"]))
nodes.append(helper.make_node("Cast", ["Zf"], ["Z"], to=F16))  # [1,1,5,30] f16

# bandsum = Conv(Z, wsel) -> [1,B,1,30] ; need f16 conv -> Conv supports f16
wsel = np.zeros((B, 1, 5, 1), np.float16); hb = np.zeros((B,), np.float16)
for i, (r0, r1, h) in enumerate(bands):
    wsel[i, 0, r0:r1 + 1, 0] = 1.0; hb[i] = h
C("wsel", wsel)
nodes.append(helper.make_node("Conv", ["Z", "wsel"], ["bandsum"], kernel_shape=[5, 1], pads=[0, 0, 0, 0]))  # f16 [1,B,1,30]
# bandZ_bool = bandsum >= hb  == Greater(bandsum, hb-0.5)
Cf16("hb_m", (hb.astype(np.float32) - 0.5).reshape(1, B, 1, 1))
nodes.append(helper.make_node("Greater", ["bandsum", "hb_m"], ["bandZ_bool"]))  # bool [1,B,1,30]
nodes.append(helper.make_node("Cast", ["bandZ_bool"], ["bandZ"], to=F16))  # f16 0/1

Cf16("zero16", 0.0)
# shift kernels: grouped Conv kernel [B,1,1,s+1] with single 1; pads keep width 30
shift_w = {}
for s in SHIFTS:
    kL = np.zeros((B, 1, 1, s + 1), np.float16); kL[:, 0, 0, 0] = 1.0   # out[c]=in[c-s] with pad left=s
    kR = np.zeros((B, 1, 1, s + 1), np.float16); kR[:, 0, 0, s] = 1.0   # out[c]=in[c+s] with pad right=s
    C(f"kL{s}", kL); C(f"kR{s}", kR)

def shift(cur, s, direction, tag):
    if direction == "L":
        nodes.append(helper.make_node("Conv", [cur, f"kL{s}"], [f"{tag}_out"],
                                      kernel_shape=[1, s + 1], pads=[0, s, 0, 0], group=B))
    else:
        nodes.append(helper.make_node("Conv", [cur, f"kR{s}"], [f"{tag}_out"],
                                      kernel_shape=[1, s + 1], pads=[0, 0, 0, s], group=B))
    return f"{tag}_out"

def doubling(base, direction):
    cur = base
    for s in SHIFTS:
        src = shift(cur, s, direction, f"sh_{direction}{s}")
        Cf16(f"sm_{direction}{s}", float(s) - 0.5)  # threshold s-0.5 -> cur>=s
        nodes.append(helper.make_node("Greater", [cur, f"sm_{direction}{s}"], [f"ge_{direction}{s}"]))
        nodes.append(helper.make_node("Where", [f"ge_{direction}{s}", src, "zero16"], [f"ext_{direction}{s}"]))
        nxt = f"cur_{direction}{s}"
        nodes.append(helper.make_node("Add", [cur, f"ext_{direction}{s}"], [nxt]))
        cur = nxt
    return cur

Lname = doubling("bandZ", "L")
Rname = doubling("bandZ", "R")

# runlen = L + R - 1 ; area = hb * runlen ; mask non-bandZ to 0
nodes.append(helper.make_node("Add", [Lname, Rname], ["LR"]))
Cf16("one16", 1.0)
nodes.append(helper.make_node("Sub", ["LR", "one16"], ["runlen"]))
Cf16("hbcol", hb.reshape(1, B, 1, 1))
nodes.append(helper.make_node("Mul", ["runlen", "hbcol"], ["area"]))  # f16 [1,B,1,30]
nodes.append(helper.make_node("Where", ["bandZ_bool", "area", "zero16"], ["areaM"]))  # f16

# A* = ReduceMax ; paint = areaM >= A* (i.e. ==) & bandZ & A*>0
nodes.append(helper.make_node("ReduceMax", ["areaM"], ["Astar"], keepdims=1))  # [1,1,1,1] f16
Cf16("half16", 0.5)
nodes.append(helper.make_node("Sub", ["Astar", "half16"], ["Astar_m"]))  # A*-0.5
nodes.append(helper.make_node("Greater", ["areaM", "Astar_m"], ["eqA"]))  # bool: areaM>=A*
nodes.append(helper.make_node("And", ["eqA", "bandZ_bool"], ["paintband"]))
nodes.append(helper.make_node("Greater", ["Astar", "half16"], ["has"]))  # A*>=1
nodes.append(helper.make_node("And", ["paintband", "has"], ["paintband2"]))
nodes.append(helper.make_node("Cast", ["paintband2"], ["pb_f"], to=F16))  # f16 [1,B,1,30]

# expand band->rows directly to 30 rows via ConvTranspose weight wexp [B,1,30,1]
wexp = np.zeros((B, 1, 30, 1), np.float16)
for i, (r0, r1, h) in enumerate(bands):
    wexp[i, 0, r0:r1 + 1, 0] = 1.0
C("wexp", wexp)
nodes.append(helper.make_node("ConvTranspose", ["pb_f", "wexp"], ["paint_f"], kernel_shape=[30, 1], pads=[0, 0, 0, 0]))  # f16 [1,1,30,30]
nodes.append(helper.make_node("Greater", ["paint_f", "half16"], ["paint_bool"]))  # bool [1,1,30,30]

six = np.zeros((1, 10, 1, 1), np.float32); six[0, 6, 0, 0] = 1.0
C("six", six)
nodes.append(helper.make_node("Where", ["paint_bool", "six", "input"], ["output"]))

graph = helper.make_graph(nodes, "task090",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])], inits)
model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
onnx.checker.check_model(model, full_check=True)
onnx.save(model, "/tmp/b9_task090.onnx")
print("saved nodes", len(nodes), "inits", len(inits))
