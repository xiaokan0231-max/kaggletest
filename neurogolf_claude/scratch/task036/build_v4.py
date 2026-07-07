"""task036 v4: single-plane uint8 neighbor count + float32 compaction (no fp16
casts). Output via 2->10 Conv expansion.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task036.onnx"
F = TensorProto.FLOAT
U8 = TensorProto.UINT8
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    inits = [
        c("cvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        c("pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)),
        c("padval", np.uint8(255)),
        c("u0", np.uint8(0)),
        c("slax", np.array([2, 3], np.int64)),
        c("iota10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        c("e0", np.eye(10, dtype=np.float32)[0].reshape(10, 1, 1, 1)),
        c("selw_shape", np.array([10, 1, 1, 1], np.int64)),
        c("iota_r", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
        c("iota_c", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
        c("one", np.float32(1.0)),
        c("ax3", np.array(3, np.int64)),
        c("sh_w", np.array([1, 1, 1, 30], np.int64)),
        c("sh_h", np.array([1, 1, 30, 1], np.int64)),
    ]
    for k, (di, dj) in enumerate(DIRS):
        inits.append(c(f"st{k}", np.array([1 + di, 1 + dj], np.int64)))
        inits.append(c(f"en{k}", np.array([31 + di, 31 + dj], np.int64)))

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    n("Conv", ["input", "cvec"], ["grid_f"])
    n("Cast", ["grid_f"], ["grid"], to=U8)
    n("Greater", ["grid", "u0"], ["nzB"])
    n("Pad", ["grid", "pads", "padval"], ["gpad"])
    eqs = []
    for k in range(8):
        n("Slice", ["gpad", f"st{k}", f"en{k}", "slax"], [f"nb{k}"])
        n("Equal", [f"nb{k}", "grid"], [f"eqB{k}"])
        n("Cast", [f"eqB{k}"], [f"eq{k}"], to=U8)
        eqs.append(f"eq{k}")
    acc = eqs[0]
    for i, e in enumerate(eqs[1:], 1):
        n("Add", [acc, e], [f"esum{i}"]); acc = f"esum{i}"
    n("Cast", ["nzB"], ["nz"], to=U8)
    n("Mul", [acc, "nz"], ["nc"])
    n("ReduceMax", ["nc"], ["gmax"], axes=[2, 3], keepdims=1)
    n("Equal", ["nc", "gmax"], ["peakB"])
    n("Cast", ["peakB"], ["peak"], to=U8)
    n("Mul", ["grid", "peak"], ["gpeak"])
    n("ReduceMax", ["gpeak"], ["cval"], axes=[2, 3], keepdims=1)
    n("Equal", ["grid", "cval"], ["MB"])
    n("Cast", ["MB"], ["M"], to=F)
    # float32 compaction
    n("ReduceMax", ["M"], ["keep_r"], axes=[3], keepdims=1)
    n("ReduceMax", ["M"], ["keep_c"], axes=[2], keepdims=1)
    n("Reshape", ["keep_r", "sh_w"], ["krw"])
    n("CumSum", ["krw", "ax3"], ["pr0"])
    n("Sub", ["pr0", "one"], ["pr"])
    n("Equal", ["iota_r", "pr"], ["ErB"])
    n("Cast", ["ErB"], ["ErF"], to=F)
    n("Mul", ["ErF", "krw"], ["Sr"])
    n("CumSum", ["keep_c", "ax3"], ["pc0"])
    n("Sub", ["pc0", "one"], ["pcw"])
    n("Reshape", ["pcw", "sh_h"], ["pc"])
    n("Reshape", ["keep_c", "sh_h"], ["kc"])
    n("Equal", ["pc", "iota_c"], ["EcB"])
    n("Cast", ["EcB"], ["EcF"], to=F)
    n("Mul", ["EcF", "kc"], ["T"])
    n("MatMul", ["Sr", "M"], ["cc"])
    n("MatMul", ["cc", "T"], ["M_comp"])
    n("ReduceMax", ["M_comp"], ["rmax"], axes=[3], keepdims=1)
    n("ReduceMax", ["M_comp"], ["cmax"], axes=[2], keepdims=1)
    n("Mul", ["rmax", "cmax"], ["G_out"])
    n("Sub", ["G_out", "M_comp"], ["bg"])
    # output assembly
    n("Cast", ["cval"], ["cval_f"], to=F)
    n("Equal", ["iota10", "cval_f"], ["selB"])
    n("Cast", ["selB"], ["selw_pre"], to=F)
    n("Reshape", ["selw_pre", "selw_shape"], ["selw"])
    n("Concat", ["e0", "selw"], ["W"], axis=1)
    n("Concat", ["bg", "M_comp"], ["pair"], axis=1)
    n("Conv", ["pair", "W"], ["output"])

    graph = helper.make_graph(
        N, "task036",
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
