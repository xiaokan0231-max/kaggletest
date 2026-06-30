"""Compact GENUINE ONNX for task120, cropped to SxS to shrink interior tensors.

Rule: nonzero cell (r,c)=v -> 8 iff its 4 orthogonal neighbors all == v (oob=0).
Else keep input color.  Output = Where(interior30, onehot(8), input).

All interior logic runs on a cropped [1,1,S,S] uint8/bool grid; interior is then
padded back to [1,1,30,30] for the Where.  S covers the max grid in this task.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def cf(name, v): return numpy_helper.from_array(np.asarray(v, np.float32), name=name)
def ci(name, v): return numpy_helper.from_array(np.asarray(v, np.int64), name=name)
def cu(name, v): return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def build(path, S=16):
    nodes, inits = [], []

    # collapse one-hot -> color grid (f32 [1,1,30,30]); crop happens via Conv? no.
    inits.append(cf("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["g30"]))   # f32 [1,1,30,30]
    # crop to SxS and cast to uint8
    inits += [ci("cs", [0, 0]), ci("ce", [S, S]), ci("cax", [2, 3])]
    nodes.append(helper.make_node("Slice", ["g30", "cs", "ce", "cax"], ["gS"]))  # f32 [1,1,S,S]
    nodes.append(helper.make_node("Cast", ["gS"], ["c"], to=U8))                 # u8 [1,1,S,S]

    # pad ring of 0 -> [1,1,S+2,S+2]; 4 directional shifts back to SxS
    inits.append(ci("pad1", [0, 0, 1, 1, 0, 0, 1, 1]))
    inits.append(cu("zero", 0))
    nodes.append(helper.make_node("Pad", ["c", "pad1", "zero"], ["cp"]))  # u8 [1,1,S+2,S+2]
    inits.append(ci("ax", [2, 3]))
    inits.append(ci("up_s", [0, 1]));  inits.append(ci("up_e", [S, S + 1]))
    inits.append(ci("dn_s", [2, 1]));  inits.append(ci("dn_e", [S + 2, S + 1]))
    inits.append(ci("lf_s", [1, 0]));  inits.append(ci("lf_e", [S + 1, S]))
    inits.append(ci("rt_s", [1, 2]));  inits.append(ci("rt_e", [S + 1, S + 2]))
    nodes.append(helper.make_node("Slice", ["cp", "up_s", "up_e", "ax"], ["up"]))
    nodes.append(helper.make_node("Slice", ["cp", "dn_s", "dn_e", "ax"], ["dn"]))
    nodes.append(helper.make_node("Slice", ["cp", "lf_s", "lf_e", "ax"], ["lf"]))
    nodes.append(helper.make_node("Slice", ["cp", "rt_s", "rt_e", "ax"], ["rt"]))

    nodes.append(helper.make_node("Equal", ["c", "up"], ["eu"]))
    nodes.append(helper.make_node("Equal", ["c", "dn"], ["ed"]))
    nodes.append(helper.make_node("Equal", ["c", "lf"], ["el"]))
    nodes.append(helper.make_node("Equal", ["c", "rt"], ["er"]))
    inits.append(cu("zg", 0))
    nodes.append(helper.make_node("Greater", ["c", "zg"], ["nz"]))   # c>0

    nodes.append(helper.make_node("And", ["eu", "ed"], ["a1"]))
    nodes.append(helper.make_node("And", ["el", "er"], ["a2"]))
    nodes.append(helper.make_node("And", ["a1", "a2"], ["a3"]))
    nodes.append(helper.make_node("And", ["a3", "nz"], ["intS"]))   # bool [1,1,S,S]

    # pad interior back to [1,1,30,30] with False
    inits.append(ci("padb", [0, 0, 0, 0, 0, 0, 30 - S, 30 - S]))
    inits.append(numpy_helper.from_array(np.asarray(False, np.bool_), "f"))
    nodes.append(helper.make_node("Pad", ["intS", "padb", "f"], ["interior"]))  # bool [1,1,30,30]

    # output = Where(interior, onehot(8), input)
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
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b16_task120.onnx", S)
