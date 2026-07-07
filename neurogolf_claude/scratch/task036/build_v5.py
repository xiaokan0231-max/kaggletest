"""task036 v5: single-plane uint8 discriminator + dynamic Slice/Pad crop.

The shape's bbox contains ONLY the shape color + background (verified, no
contamination), so cropping the full input one-hot to the bbox and padding back
to 30x30 IS the answer. This replaces the MatMul compaction + Conv expansion
with a tiny dynamic Slice followed by a dynamic Pad.

shape color = color of the cell with the max same-color 8-neighbor count.
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
        c("cvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)),
        c("pads_g", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64)),
        c("padval", np.uint8(255)),
        c("u0", np.uint8(0)),
        c("slax", np.array([2, 3], np.int64)),
        c("crop_axes", np.array([2, 3], np.int64)),
        c("zero_f", np.float32(0.0)),
        # iota over rows/cols as int64 for bbox extents
        c("iota_r", np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)),
        c("iota_c", np.arange(30, dtype=np.int64).reshape(1, 1, 1, 30)),
        c("BIG", np.int64(99)),
        c("one_i", np.int64(1)),
        c("thirty", np.int64(30)),
        c("zero4", np.array([0, 0, 0, 0], np.int64)),
        c("zero2", np.array([0, 0], np.int64)),
    ]
    for k, (di, dj) in enumerate(DIRS):
        inits.append(c(f"st{k}", np.array([1 + di, 1 + dj], np.int64)))
        inits.append(c(f"en{k}", np.array([31 + di, 31 + dj], np.int64)))

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # 1. colored single-plane grid (uint8)
    n("Conv", ["input", "cvec"], ["grid_f"])
    n("Cast", ["grid_f"], ["grid"], to=U8)
    n("Greater", ["grid", "u0"], ["nzB"])
    n("Pad", ["grid", "pads_g", "padval"], ["gpad"])
    # 2. 8-neighbor same-color count (uint8)
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
    # 3. peak + shape color
    n("ReduceMax", ["nc"], ["gmax"], axes=[2, 3], keepdims=1)
    n("Equal", ["nc", "gmax"], ["peakB"])
    n("Cast", ["peakB"], ["peak"], to=U8)
    n("Mul", ["grid", "peak"], ["gpeak"])
    n("ReduceMax", ["gpeak"], ["cval"], axes=[2, 3], keepdims=1)   # [1,1,1,1] uint8
    # 4. shape mask -> bbox extents
    n("Equal", ["grid", "cval"], ["MB"])                           # bool [1,1,30,30]
    n("Cast", ["MB"], ["Mi"], to=I64)                              # int64 0/1
    n("ReduceMax", ["Mi"], ["keep_r"], axes=[3], keepdims=1)        # [1,1,30,1]
    n("ReduceMax", ["Mi"], ["keep_c"], axes=[2], keepdims=1)        # [1,1,1,30]
    # rmin = min row with keep; rmax = max row with keep
    n("Mul", ["iota_r", "keep_r"], ["rk"])                         # row idx where kept else 0
    n("ReduceMax", ["rk"], ["rmax"], axes=[2, 3], keepdims=1)       # scalar
    # for min: set non-kept rows to BIG
    n("Sub", ["one_i", "keep_r"], ["nkr"])                        # 1 where not kept
    n("Mul", ["nkr", "BIG"], ["nkrB"])
    n("Add", ["rk", "nkrB"], ["rkm"])
    n("ReduceMin", ["rkm"], ["rmin"], axes=[2, 3], keepdims=1)
    n("Mul", ["iota_c", "keep_c"], ["ck"])
    n("ReduceMax", ["ck"], ["cmax"], axes=[2, 3], keepdims=1)
    n("Sub", ["one_i", "keep_c"], ["nkc"])
    n("Mul", ["nkc", "BIG"], ["nkcB"])
    n("Add", ["ck", "nkcB"], ["ckm"])
    n("ReduceMin", ["ckm"], ["cmin"], axes=[2, 3], keepdims=1)
    # build starts/ends [2] int64
    n("Reshape", ["rmin", "one_dim"], ["rmin1"])
    n("Reshape", ["cmin", "one_dim"], ["cmin1"])
    n("Concat", ["rmin1", "cmin1"], ["starts"], axis=0)            # [2]
    n("Add", ["rmax", "one_i"], ["rmax1"])
    n("Add", ["cmax", "one_i"], ["cmax1"])
    n("Reshape", ["rmax1", "one_dim"], ["rmax1d"])
    n("Reshape", ["cmax1", "one_dim"], ["cmax1d"])
    n("Concat", ["rmax1d", "cmax1d"], ["ends"], axis=0)           # [2]
    # 5. crop input to bbox (tiny), pad back to 30x30
    n("Slice", ["input", "starts", "ends", "crop_axes"], ["crop"])  # [1,10,h,w]
    # pad-after amounts: h1 = 30-(rmax+1-rmin), w1 = 30-(cmax+1-cmin)
    n("Sub", ["rmax1", "rmin"], ["h"])
    n("Sub", ["cmax1", "cmin"], ["w"])
    n("Sub", ["thirty", "h"], ["h1"])
    n("Sub", ["thirty", "w"], ["w1"])
    n("Reshape", ["h1", "one_dim"], ["h1d"])
    n("Reshape", ["w1", "one_dim"], ["w1d"])
    # pads = [b0,c0,h0,w0, b1,c1,h1,w1] = [0,0,0,0, 0,0,h1,w1]
    n("Concat", ["zero4", "zero2", "h1d", "w1d"], ["pads_out"], axis=0)  # [8]
    n("Pad", ["crop", "pads_out", "zero_f"], ["output"])

    inits.append(c("one_dim", np.array([1], np.int64)))

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
