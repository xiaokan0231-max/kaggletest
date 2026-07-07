"""Build compact ONNX for task277.

Rule (verified 266/266): input has exactly 3 connected objects (8-conn) of color 8.
The two largest are equal-sized; the unique SMALLEST object is recolored 2, the
other two are recolored 1. Background stays 0. Grids 10x10 in a 30x30 canvas.

We discriminate the smallest object by bounding-box span (h-1)+(w-1), which is
strictly minimal for the odd object on all 266 cases (verified). Bounding-box
extents are computed by a cheap 8-connected MAX/MIN flood of the row/col index
restricted to the object mask -- every interior tensor stays [1,<=4,10,10].

Pipeline:
 1. collapse one-hot input -> color grid g30 [1,1,30,30] f32; Slice top-left 10x10.
 2. mask m=(g==8).
 3. 4-channel flood of [row, -row, col, -col] (NEG sentinel off-mask) -> per-object
    max-row, max(-row)=-min-row, max-col, max(-col)=-min-col. 8 iterations.
 4. span = (maxr-minr)+(maxc-minc) on mask; sbig=ReduceMax(span).
 5. min object = mask & 0<span<sbig. color grid gm = 1 on mask + 1 on min-object -> {0,1,2}.
 6. Pad to 30x30 sentinel 255; output = Equal(gm30, colors[0..9]) (graph output, excluded).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
S = 10
NITER = 8
NEG = -100.0


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # 1. color grid -> slice 10x10
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("ca", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "ca"], ["g"]))  # f32 [1,1,10,10]

    # 2. mask m=(g==8) f16
    inits.append(cf("c8", 8.0))
    nodes.append(helper.make_node("Equal", ["g", "c8"], ["mb"]))   # bool [1,1,10,10]
    nodes.append(helper.make_node("Cast", ["mb"], ["m"], to=F16))  # f16 [1,1,10,10]

    # 3. 4-channel coordinate seed: [row, -row, col, -col] on mask, NEG off mask.
    rr = (np.arange(S).reshape(S, 1) * np.ones((1, S))).astype(np.float16)
    cc = (np.ones((S, 1)) * np.arange(S).reshape(1, S)).astype(np.float16)
    coord = np.stack([rr, -rr, cc, -cc], 0).reshape(1, 4, S, S)  # [1,4,10,10]
    inits.append(numpy_helper.from_array(coord.astype(np.float16), "coord"))
    inits.append(c16("neg", NEG))
    # seed = where(m, coord, NEG). m is [1,1,10,10] broadcasts over 4 chans.
    # build mask4 bool broadcast: use Where with mb (bool [1,1,10,10]) broadcasting.
    nodes.append(helper.make_node("Where", ["mb", "coord", "neg"], ["F0"]))  # [1,4,10,10] f16
    cur = "F0"
    for k in range(NITER):
        nodes.append(helper.make_node("MaxPool", [cur], [f"Fp{k}"],
                                      kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
        # re-mask: where(mb, Fp, NEG)
        nodes.append(helper.make_node("Where", ["mb", f"Fp{k}", "neg"], [f"F{k+1}"]))
        cur = f"F{k+1}"
    Ffin = cur  # [1,4,10,10] : ch0=maxrow ch1=-minrow ch2=maxcol ch3=-mincol

    # 4. span = (maxrow - minrow) + (maxcol - mincol)
    #        = (ch0 + ch1) + (ch2 + ch3)   [since ch1=-minrow, ch3=-mincol]
    # ReduceSum over channel axis gives ch0+ch1+ch2+ch3 directly.
    inits.append(ci("ax1", [1]))
    nodes.append(helper.make_node("ReduceSum", [Ffin, "ax1"], ["spanraw"], keepdims=1))  # [1,1,10,10]
    # restrict to mask (off-mask is 4*NEG, kill it)
    nodes.append(helper.make_node("Mul", ["spanraw", "m"], ["span"]))  # f16 [1,1,10,10], 0 off mask

    # 5. sbig = max span; min object = 0<span<sbig
    nodes.append(helper.make_node("ReduceMax", ["span"], ["sbig"], axes=[1, 2, 3], keepdims=1))
    inits.append(c16("half", 0.5))
    nodes.append(helper.make_node("Sub", ["sbig", "half"], ["sbigm"]))      # sbig-0.5
    nodes.append(helper.make_node("Less", ["span", "sbigm"], ["ltb"]))      # span<sbig (bg true)
    nodes.append(helper.make_node("And", ["ltb", "mb"], ["ismin"]))        # min-object cells
    nodes.append(helper.make_node("Cast", ["ismin"], ["isminf"], to=F16))
    nodes.append(helper.make_node("Add", ["m", "isminf"], ["gmf"]))         # 0/1/2
    nodes.append(helper.make_node("Cast", ["gmf"], ["gmu"], to=U8))

    # 6. pad + Equal -> output
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(255, np.uint8), "sent"))
    nodes.append(helper.make_node("Pad", ["gmu", "pads", "sent"], ["gm30"]))  # u8 [1,1,30,30]
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm30", "colors"], ["output"]))   # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t277", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b13_task277.onnx")
