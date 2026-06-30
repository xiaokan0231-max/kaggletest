"""Build compact ONNX for task196.

Rule (verified 262/262 arc-gen + 4/4 train+test): recolor blue (1) shapes to
green (3) iff the shape is a closed ring that ENCLOSES at least one background
cell (a hole). Shapes with no enclosed hole stay blue.

Algorithm (genuine, no labeling/Loop):
 1. Collapse one-hot input -> color grid g [1,1,30,30] f32 (Conv weight 0..9).
 2. Crop to 16x16 (all grids <=15, placed top-left). bg=(g==0), one=(g==1) bool.
 3. Border-connected background via bounded flood: seed reach=border bg cells,
    iterate reach = MaxPool3x3(reach) AND bg  (ITERS times, 8-connectivity).
    Cropping to 16x16 removes the long outer 'sea' corridor so ITERS=12 (>=10
    needed) converges on all cases.
 4. enclosed = bg AND NOT reach  (the holes).
 5. ring = one AND MaxPool3x3(enclosed)  (the hollow-rect border around holes).
 6. gm = where(ring, 3, g) as uint8; pad to 30x30 with sentinel 255.
 7. output one-hot = Equal(gm30, colors[0..9]) -> graph output (excluded mem).
All interior tensors are bool/uint8 [1,1,16,16] (256B) -> tiny memory.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
S = 15
ITERS = 12


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path, iters=ITERS):
    nodes, inits = [], []

    # 1) Slice channel 1 (the blue==1 mask) cropped to SxS in ONE op, directly from
    #    the (excluded) input. Output blue15 [1,1,S,S] f32 -> only ~900B, avoiding any
    #    [1,1,30,30] color grid. Non-blue cells (bg AND padding) are all 0 here, which
    #    is exactly the "passable" set for the flood.
    inits.append(ci("cs", [1, 0, 0]))      # channel 1, h 0, w 0
    inits.append(ci("ce", [2, S, S]))      # channel 2 (excl), h S, w S
    inits.append(ci("ca", [1, 2, 3]))      # axes: channel, height, width
    nodes.append(helper.make_node("Slice", ["input", "cs", "ce", "ca"], ["blue15"]))

    # 2) masks: one = (blue15 > 0.5) blue cells; passable = NOT blue (background AND
    #    padding are both passable, so the open sea connects to the border). Pattern-b
    #    Where-as-output keeps the original input for every non-ring cell, so no
    #    validity mask is needed.
    inits.append(cf("half", 0.5))
    inits.append(cu("c0", 0))
    nodes.append(helper.make_node("Greater", ["blue15", "half"], ["one"]))  # bool (blue)
    nodes.append(helper.make_node("Not", ["one"], ["passb"]))               # bool (non-blue)
    nodes.append(helper.make_node("Cast", ["passb"], ["bgu"], to=U8))       # u8 0/1

    # 3) seed reach = border bg. border const u8 (1 on frame). reach0 = Min(bgu,border).
    border = np.zeros((1, 1, S, S), np.uint8)
    border[:, :, 0, :] = 1
    border[:, :, -1, :] = 1
    border[:, :, :, 0] = 1
    border[:, :, :, -1] = 1
    inits.append(numpy_helper.from_array(border, "borderc"))
    nodes.append(helper.make_node("Min", ["bgu", "borderc"], ["reach0"]))  # u8

    # flood: reach = Min(MaxPool3x3(reach), bgu)   (all uint8, 256B each)
    cur = "reach0"
    for i in range(iters):
        mpn = f"mp{i}"
        rn = f"reach{i+1}"
        nodes.append(helper.make_node("MaxPool", [cur], [mpn],
                                      kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Min", [mpn, "bgu"], [rn]))  # AND with bg
        cur = rn

    # 4) enclosed = passable AND (reach == 0).  Equal(reach_u8,0) -> bool directly.
    #    (padding is always border-reachable, so enclosed = real interior holes.)
    nodes.append(helper.make_node("Equal", [cur, "c0"], ["notreach"]))  # bool reach==0
    nodes.append(helper.make_node("And", ["passb", "notreach"], ["enclosed"]))  # bool

    # 5) ring = one AND MaxPool3x3(enclosed).  enclosed bool -> u8, dilate, -> bool.
    nodes.append(helper.make_node("Cast", ["enclosed"], ["enclu"], to=U8))
    nodes.append(helper.make_node("MaxPool", ["enclu"], ["encld"],
                                  kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))  # u8
    nodes.append(helper.make_node("Cast", ["encld"], ["encldb"], to=BOOL))
    nodes.append(helper.make_node("And", ["one", "encldb"], ["ring"]))  # bool

    # 6) pad ring to 30x30 (False outside crop) -> ring30 bool [1,1,30,30].
    inits.append(ci("pads", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "padfalse"))
    nodes.append(helper.make_node("Pad", ["ring", "pads", "padfalse"], ["ring30"]))  # bool

    # 7) Where-as-output (pattern b): keep input everywhere except ring cells, which
    #    become the green one-hot e3=[0,0,0,1,0..]. The only [1,10,30,30] tensor is
    #    the graph output (excluded). All non-ring cells (bg/blue/padding) keep their
    #    original one-hot from input, so validity is handled for free.
    e3 = np.zeros((1, 10, 1, 1), np.float32)
    e3[0, 3, 0, 0] = 1.0
    inits.append(cf("e3", e3))
    nodes.append(helper.make_node("Where", ["ring30", "e3", "input"], ["output"]))  # f32 [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t196", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes), "iters", iters)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/b7_task196.onnx"
    it = int(sys.argv[2]) if len(sys.argv) > 2 else ITERS
    build(p, it)
