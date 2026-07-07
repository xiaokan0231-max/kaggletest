"""task120, minimal genuine build via 3x3 uniform-neighborhood detection.

For this task every nonzero region is a SOLID rectangle, so the exact rule
"cell -> 8 iff 4 orthogonal neighbors == self" is equivalent to
"cell -> 8 iff the full 3x3 neighborhood is a single nonzero color"
(an interior cell of a solid blob).  We detect that with a 3x3 max==min==self.

Pipeline (cropped SxS, float16 grid):
  g30 f32 -> crop -> f16 gS -> mx=MaxPool3x3(gS), mn=-MaxPool3x3(-gS)
  uniform = (mx==mn) & (gS>0)  -> pad to 30x30 -> Where(., onehot8, input).
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL


def cf(name, v): return numpy_helper.from_array(np.asarray(v, np.float32), name=name)
def c16(name, v): return numpy_helper.from_array(np.asarray(v, np.float16), name=name)
def ci(name, v): return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path, S=16):
    nodes, inits = [], []

    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))   # f32 [1,1,30,30]
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("cax", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "cax"], ["gSf"]))
    nodes.append(helper.make_node("Cast", ["gSf"], ["g"], to=F16))       # f16 [1,1,S,S]

    # explicit 0-pad ring so border windows DO see the zero outside (MaxPool's own
    # padding fills with -inf and would be ignored -> wrong).
    inits.append(ci("pad1", [0, 0, 1, 1, 0, 0, 1, 1]))
    inits.append(c16("zp", 0.0))
    nodes.append(helper.make_node("Pad", ["g", "pad1", "zp"], ["gp"]))   # f16 [1,1,S+2,S+2]
    nodes.append(helper.make_node("MaxPool", ["gp"], ["mx"],
                                  kernel_shape=[3, 3], strides=[1, 1]))   # [1,1,S,S]
    nodes.append(helper.make_node("Neg", ["gp"], ["ng"]))
    nodes.append(helper.make_node("MaxPool", ["ng"], ["nmn"],
                                  kernel_shape=[3, 3], strides=[1, 1]))   # = -min over 0-padded window
    nodes.append(helper.make_node("Add", ["mx", "nmn"], ["rng"]))  # max - min
    # uniform nonzero: range==0 AND center>0  (border cells have a 0 neighbor -> range>=center>0)
    inits.append(c16("z", 0.0))
    inits.append(c16("half", 0.5))
    nodes.append(helper.make_node("Less", ["rng", "half"], ["uni"]))   # range==0
    nodes.append(helper.make_node("Greater", ["g", "z"], ["nz"]))      # center>0
    nodes.append(helper.make_node("And", ["uni", "nz"], ["intS"]))     # bool [1,1,S,S]

    inits.append(ci("padb", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "f"))
    nodes.append(helper.make_node("Pad", ["intS", "padb", "f"], ["interior"]))  # bool [1,1,30,30]

    e8 = np.zeros((1, 10, 1, 1), np.float32); e8[0, 8, 0, 0] = 1.0
    inits.append(cf("e8", e8))
    nodes.append(helper.make_node("Where", ["interior", "e8", "input"], ["output"]))

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "t120", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes), "S", S)


if __name__ == "__main__":
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b16c_task120.onnx", S)
