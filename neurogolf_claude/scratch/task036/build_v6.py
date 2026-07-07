"""task036 v6: same algorithm as v5 (single-plane 8-neighbor discriminator +
dynamic Slice/Pad crop) but minimized node/init count and short names to shrink
the serialized file (memory falls back to file size for the data-dependent crop).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task036.onnx"
F = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    inits = [
        c("cv", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        c("pg", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)),
        c("pv", np.uint8(255)),
        c("u0", np.uint8(0)),
        c("sa", np.array([2, 3], np.int64)),
        c("zf", np.float32(0.0)),
        c("ir", np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)),
        c("ic", np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)),
        c("BG", np.int64(99)),
        c("o1", np.int64(1)),
        c("T30", np.int64(30)),
        c("z4", np.array([0, 0, 0, 0], np.int64)),
        c("z2", np.array([0, 0], np.int64)),
        c("d1", np.array([1], np.int64)),
        c("ax1", np.array([1], np.int64)),
    ]
    for k, (di, dj) in enumerate(DIRS):
        inits.append(c(f"s{k}", np.array([1 + di, 1 + dj], np.int64)))
        inits.append(c(f"e{k}", np.array([31 + di, 31 + dj], np.int64)))

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    n("Conv", ["input", "cv"], ["gf"])
    n("Cast", ["gf"], ["g"], to=U8)
    n("Greater", ["g", "u0"], ["nzb"])
    n("Pad", ["g", "pg", "pv"], ["gp"])
    # 8 directional equality masks summed in uint8
    eqs = []
    for k in range(8):
        n("Slice", ["gp", f"s{k}", f"e{k}", "sa"], [f"b{k}"])
        n("Equal", [f"b{k}", "g"], [f"qb{k}"])
        n("Cast", [f"qb{k}"], [f"q{k}"], to=U8)
        eqs.append(f"q{k}")
    acc = eqs[0]
    for i, e in enumerate(eqs[1:], 1):
        n("Add", [acc, e], [f"sm{i}"]); acc = f"sm{i}"
    n("Cast", ["nzb"], ["nz"], to=U8)
    n("Mul", [acc, "nz"], ["nc"])
    n("ReduceMax", ["nc"], ["gm"], axes=[2, 3], keepdims=1)
    n("Equal", ["nc", "gm"], ["pkb"])
    n("Cast", ["pkb"], ["pk"], to=U8)
    n("Mul", ["g", "pk"], ["gpk"])
    n("ReduceMax", ["gpk"], ["cval"], axes=[2, 3], keepdims=1)
    # bbox extents
    n("Equal", ["g", "cval"], ["mb"])
    n("Cast", ["mb"], ["mi"], to=I64)
    n("ReduceMax", ["mi"], ["kr"], axes=[3], keepdims=1)
    n("ReduceMax", ["mi"], ["kc"], axes=[2], keepdims=1)
    n("Mul", ["ir", "kr"], ["rk"])
    n("ReduceMax", ["rk"], ["rmx"], axes=[2, 3], keepdims=1)
    n("Sub", ["o1", "kr"], ["nkr"])
    n("Mul", ["nkr", "BG"], ["nkrb"])
    n("Add", ["rk", "nkrb"], ["rkm"])
    n("ReduceMin", ["rkm"], ["rmn"], axes=[2, 3], keepdims=1)
    n("Mul", ["ic", "kc"], ["ck"])
    n("ReduceMax", ["ck"], ["cmx"], axes=[2, 3], keepdims=1)
    n("Sub", ["o1", "kc"], ["nkc"])
    n("Mul", ["nkc", "BG"], ["nkcb"])
    n("Add", ["ck", "nkcb"], ["ckm"])
    n("ReduceMin", ["ckm"], ["cmn"], axes=[2, 3], keepdims=1)
    n("Reshape", ["rmn", "d1"], ["rmn1"])
    n("Reshape", ["cmn", "d1"], ["cmn1"])
    n("Concat", ["rmn1", "cmn1"], ["st"], axis=0)
    n("Add", ["rmx", "o1"], ["rmx1"])
    n("Add", ["cmx", "o1"], ["cmx1"])
    n("Reshape", ["rmx1", "d1"], ["rmx1d"])
    n("Reshape", ["cmx1", "d1"], ["cmx1d"])
    n("Concat", ["rmx1d", "cmx1d"], ["en"], axis=0)
    n("Slice", ["input", "st", "en", "sa"], ["crop"])
    n("Sub", ["rmx1", "rmn"], ["h"])
    n("Sub", ["cmx1", "cmn"], ["w"])
    n("Sub", ["T30", "h"], ["h1"])
    n("Sub", ["T30", "w"], ["w1"])
    n("Reshape", ["h1", "d1"], ["h1d"])
    n("Reshape", ["w1", "d1"], ["w1d"])
    n("Concat", ["z4", "z2", "h1d", "w1d"], ["po"], axis=0)
    n("Pad", ["crop", "po", "zf"], ["output"])

    graph = helper.make_graph(
        N, "t",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT, len(model.SerializeToString()), "bytes")


if __name__ == "__main__":
    build()
