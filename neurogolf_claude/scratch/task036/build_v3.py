"""task036 v3: single-plane pipeline, uint8 neighbor-count, custom fp16 compaction.

Same algorithm as v2 but:
- colored grid + neighbor counts done in uint8 (1 byte) not float32 (4 bytes).
- compaction permutation matrices in float16 (2 bytes).
- minimal tensor count.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task036.onnx"
F = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
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
        # fp16 compaction constants
        c("iota_r16", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)),
        c("iota_c16", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)),
        c("one16", np.float16(1.0)),
        c("ax3", np.array(3, np.int64)),
        c("sh_w16", np.array([1, 1, 1, 30], np.int64)),
        c("sh_h16", np.array([1, 1, 30, 1], np.int64)),
    ]
    for k, (di, dj) in enumerate(DIRS):
        inits.append(c(f"st{k}", np.array([1 + di, 1 + dj], np.int64)))
        inits.append(c(f"en{k}", np.array([31 + di, 31 + dj], np.int64)))

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # 1. colored grid (Conv outputs float) -> cast to uint8
    n("Conv", ["input", "cvec"], ["grid_f"])                     # [1,1,30,30] float
    n("Cast", ["grid_f"], ["grid"], to=U8)                       # uint8 colors 0..9
    n("Greater", ["grid", "u0"], ["nzB"])                        # bool nonzero
    # 2. pad with sentinel 255
    n("Pad", ["grid", "pads", "padval"], ["gpad"])               # [1,1,32,32] uint8
    # 3. neighbor equality counts (uint8)
    eqs = []
    for k in range(8):
        n("Slice", ["gpad", f"st{k}", f"en{k}", "slax"], [f"nb{k}"])
        n("Equal", [f"nb{k}", "grid"], [f"eqB{k}"])
        n("Cast", [f"eqB{k}"], [f"eq{k}"], to=U8)
        eqs.append(f"eq{k}")
    acc = eqs[0]
    for i, e in enumerate(eqs[1:], 1):
        out = f"esum{i}"
        n("Add", [acc, e], [out]); acc = out
    # mask out background-cell counts: nc = acc where nonzero else 0
    n("Cast", ["nzB"], ["nz"], to=U8)
    n("Mul", [acc, "nz"], ["nc"])                                # [1,1,30,30] uint8
    # 4. peak + shape color
    n("ReduceMax", ["nc"], ["gmax"], axes=[2, 3], keepdims=1)     # [1,1,1,1] uint8
    n("Equal", ["nc", "gmax"], ["peakB"])
    n("Cast", ["peakB"], ["peak"], to=U8)
    n("Mul", ["grid", "peak"], ["gpeak"])                        # uint8: color at peak cells
    n("ReduceMax", ["gpeak"], ["cval"], axes=[2, 3], keepdims=1)  # [1,1,1,1] uint8 shape color
    # 5. shape mask (uint8 -> float16 for compaction)
    n("Equal", ["grid", "cval"], ["MB"])
    n("Cast", ["MB"], ["M"], to=F16)                             # [1,1,30,30] f16
    # 6. compaction (fp16) -- inline (cropkit but fp16)
    n("ReduceMax", ["M"], ["keep_r"], axes=[3], keepdims=1)       # [1,1,30,1]
    n("ReduceMax", ["M"], ["keep_c"], axes=[2], keepdims=1)       # [1,1,1,30]
    # Sr: gather kept rows to top
    n("Reshape", ["keep_r", "sh_w16"], ["krw"])                  # [1,1,1,30]
    n("CumSum", ["krw", "ax3"], ["pr0"])
    n("Sub", ["pr0", "one16"], ["pr"])
    n("Equal", ["iota_r16", "pr"], ["ErB"])
    n("Cast", ["ErB"], ["ErF"], to=F16)
    n("Mul", ["ErF", "krw"], ["Sr"])                            # [1,1,30,30] f16
    # T: gather kept cols to left
    n("CumSum", ["keep_c", "ax3"], ["pc0"])
    n("Sub", ["pc0", "one16"], ["pcw"])
    n("Reshape", ["pcw", "sh_h16"], ["pc"])
    n("Reshape", ["keep_c", "sh_h16"], ["kc"])
    n("Equal", ["pc", "iota_c16"], ["EcB"])
    n("Cast", ["EcB"], ["EcF"], to=F16)
    n("Mul", ["EcF", "kc"], ["T"])                              # [1,1,30,30] f16
    n("MatMul", ["Sr", "M"], ["cc"])
    n("MatMul", ["cc", "T"], ["M_comp"])                        # [1,1,30,30] f16
    # output-grid bbox mask
    n("ReduceMax", ["M_comp"], ["rmax"], axes=[3], keepdims=1)
    n("ReduceMax", ["M_comp"], ["cmax"], axes=[2], keepdims=1)
    n("Mul", ["rmax", "cmax"], ["G_out"])
    n("Sub", ["G_out", "M_comp"], ["bg"])                       # f16
    # 7. assemble one-hot output via Conv (float)
    n("Cast", ["cval"], ["cval_f"], to=F)
    n("Equal", ["iota10", "cval_f"], ["selB"])                  # [1,10,1,1]
    n("Cast", ["selB"], ["selw_pre"], to=F)
    n("Reshape", ["selw_pre", "selw_shape"], ["selw"])          # [10,1,1,1]
    n("Concat", ["e0", "selw"], ["W"], axis=1)                  # [10,2,1,1]
    n("Cast", ["bg"], ["bg_f"], to=F)
    n("Cast", ["M_comp"], ["Mc_f"], to=F)
    n("Concat", ["bg_f", "Mc_f"], ["pair"], axis=1)            # [1,2,30,30]
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
