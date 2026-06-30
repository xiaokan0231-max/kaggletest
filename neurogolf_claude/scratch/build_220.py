"""Build compact ONNX for task220.

Rule (verified 267/267): every nonzero pixel (color in {2,3,8}) is surrounded by a
3x3 frame in a mapped color (2->1, 3->6, 8->4). The frame fills only empty (0)
cells; original centers are preserved. Frames never overlap (min center Chebyshev
distance = 3) and are clipped at the grid boundary.

Strategy (full 30x30, no crop -- grids vary 6..16 and hidden test may be larger):
  - g  = color grid [1,1,30,30] f32 via Conv weight [0..9]  (the 3600B floor).
  - valid = in-grid mask via Conv weight all-ones over the 10 channels (1 in-grid,
    0 out-of-grid). Frames must not be drawn outside the real grid.
  - For each center color v: mask_v = (g==v); dilate with MaxPool 3x3; the frame
    region is dilate(mask_v) AND NOT mask_v.  Since frames are disjoint we just sum
    framecolor_v * frame_region_v.
  - gm = g + framevals (centers stay, frames land on former zeros), clamped to valid.
  - Pad out-of-grid cells to sentinel 255, then Equal(gm, colors[0..9]) IS the
    graph output -> no interior [1,10,*] tensor.

All interior spatial tensors are uint8/bool [1,1,30,30] (900 B each).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL

CENTERS = [2, 3, 8]
FRAME = {2: 1, 3: 6, 8: 4}


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # ---- color grid g [1,1,30,30] f32 (input excluded) ----
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g32"]))
    nodes.append(helper.make_node("Cast", ["g32"], ["g"], to=F16))  # f16 [1,1,30,30]

    # ---- valid (in-grid) mask: sum over 10 channels of input ----
    inits.append(cf("Wone", np.ones((1, 10, 1, 1), dtype=np.float32)))
    nodes.append(helper.make_node("Conv", ["input", "Wone"], ["vsum32"]))  # f32 1/0
    nodes.append(helper.make_node("Cast", ["vsum32"], ["valid"], to=F16))  # 1.0 in-grid, 0 out

    # ---- per-center frame regions, summed weighted by frame color ----
    inits.append(c16("half", 0.5))
    frame_terms = []
    for v in CENTERS:
        inits.append(c16(f"c{v}", float(v)))
        nodes.append(helper.make_node("Equal", ["g", f"c{v}"], [f"m{v}b"]))      # bool center mask
        nodes.append(helper.make_node("Cast", [f"m{v}b"], [f"m{v}"], to=F16))    # f16 0/1
        # dilate 3x3 (frame footprint incl center)
        nodes.append(helper.make_node("MaxPool", [f"m{v}"], [f"d{v}"],
                                      kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]))
        # frame ring only = dilate AND NOT center  ->  d - m  (both 0/1, disjoint subtract ok)
        nodes.append(helper.make_node("Sub", [f"d{v}", f"m{v}"], [f"ring{v}"]))  # f16 0/1
        # weight by frame color
        inits.append(c16(f"fc{v}", float(FRAME[v])))
        nodes.append(helper.make_node("Mul", [f"ring{v}", f"fc{v}"], [f"fv{v}"]))
        frame_terms.append(f"fv{v}")

    # sum frame contributions (disjoint regions)
    nodes.append(helper.make_node("Add", [frame_terms[0], frame_terms[1]], ["fa"]))
    nodes.append(helper.make_node("Add", ["fa", frame_terms[2]], ["framevals"]))  # f16

    # gm = g + framevals (centers preserved, frames on former zeros)
    nodes.append(helper.make_node("Add", ["g", "framevals"], ["gmf"]))
    # restrict to valid cells: multiply by valid (out-of-grid -> 0, harmless before sentinel)
    nodes.append(helper.make_node("Mul", ["gmf", "valid"], ["gmv"]))  # f16

    # cast to uint8 (values are exact integers 0..9)
    nodes.append(helper.make_node("Cast", ["gmv"], ["gmu"], to=U8))   # u8 [1,1,30,30]

    # out-of-grid cells -> sentinel 255 so no color matches (Equal all-false -> padding 10)
    # validu8: 1 in-grid, 0 out. sentinel = gmu*valid + 255*(1-valid)
    nodes.append(helper.make_node("Cast", ["valid"], ["validu"], to=U8))  # u8 0/1
    # where(validu==1, gmu, 255)
    nodes.append(helper.make_node("Cast", ["validu"], ["validb"], to=BOOL))
    inits.append(numpy_helper.from_array(np.asarray(255, np.uint8), "sent"))
    nodes.append(helper.make_node("Where", ["validb", "gmu", "sent"], ["gm"]))  # u8 [1,1,30,30]

    # final one-hot: output[c] = (gm == c). Equal IS the graph output (excluded).
    inits.append(numpy_helper.from_array(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "colors"))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))  # bool [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t220", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b17_task220.onnx")
