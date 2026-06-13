"""task020: D4 (dihedral-8) symmetrization of the figure about its square bbox.

Rule (verified on all 266 train+test+arc-gen): the input contains one figure
whose nonzero bounding box is always 5x5 (square). The output is the union (max
over color channels' spatial support) of the 8 dihedral transforms (identity,
flipud, fliplr, rot180, and the 4 transpose-family) applied about the figure's
own bbox. Colors are preserved (each transform is a pure spatial permutation
applied identically to every one-hot channel).

Construction (all in the 30x30 padded one-hot space, [1,10,30,30] float):
  presence M = max over channels                       -> [1,1,30,30]
  row-presence rp = max over cols, col-presence cp     -> projections
  y0,y1 = min/max row index with rp=1 ; x0,x1 likewise
  sy=y0+y1, sx=x0+x1, off=y0-x0
  Pr[i,r]=1 iff i+r==sy   (row reflection about bbox row-center)
  Pc[i,c]=1 iff i+c==sx   (col reflection about bbox col-center)
  Sr[i,r]=1 iff i-r==off  (row shift +off for the offset transpose)
  Sc[r,j]=1 iff r-j==off  (col shift -off)  == Sr  (same matrix)
  Xt = transpose(X) over (H,W)
  Tg = Sr @ Xt @ Sr
  out = max(X, Pr@X, X@Pc, Pr@X@Pc, Tg, Pr@Tg, Tg@Pc, Pr@Tg@Pc)

Everything is permutation MatMuls + elementwise Max; no Loop/Scan/If, standard
opset. Intermediate tensors are small (mostly [1,1,30,30] masks and a handful of
[1,10,30,30] grids), far below the fat 14531-byte model.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf2_task020.onnx"


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(path=OUT):
    H = 30
    iota = np.arange(H, dtype=np.float32)
    inits = [
        c("iota_r", iota.reshape(1, 1, H, 1)),       # [1,1,30,1]
        c("iota_c", iota.reshape(1, 1, 1, H)),       # [1,1,1,30]
        c("big", np.float32(1e4)),
        c("A2d", iota.reshape(H, 1)),                # [30,1] for outer sums
        c("B2d", iota.reshape(1, H)),                # [1,30]
    ]
    N = []
    n = lambda op, i, o, **k: N.append(helper.make_node(op, i, o, **k))

    # presence mask over channels (use ReduceMax: any color present)
    # figure-only grid: zero out background channel 0 so Max-union is clean.
    inits.append(c("cmask", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)))
    inits.append(c("e0", np.array([1] + [0] * 9, np.float32).reshape(1, 10, 1, 1)))
    n("Mul", ["input", "cmask"], ["Xf"])                              # [1,10,30,30] ch0=0
    # presence over figure channels (1 where any non-bg color present)
    n("ReduceMax", ["Xf"], ["M"], axes=[1], keepdims=1)              # [1,1,30,30]
    # row presence rp [1,1,30,1] (max over cols); col presence cp [1,1,1,30]
    n("ReduceMax", ["M"], ["rp"], axes=[3], keepdims=1)               # [1,1,30,1]
    n("ReduceMax", ["M"], ["cp"], axes=[2], keepdims=1)               # [1,1,1,30]
    # y0=min row idx where rp>0 ; y1=max. rp is already 0/1 (one-hot channel max).
    # rows_for_min = iota_r*rp + (1-rp)*big  -> present rows keep iota, absent -> big
    inits.append(c("one", np.float32(1.0)))
    n("Mul", ["iota_r", "rp"], ["irp"])
    n("Sub", ["one", "rp"], ["inv_rp"])               # [1,1,30,1]
    n("Mul", ["inv_rp", "big"], ["bigrp"])
    n("Add", ["irp", "bigrp"], ["rmin_src"])          # iota where present else big
    n("ReduceMin", ["rmin_src"], ["y0"], axes=[2], keepdims=1)   # [1,1,1,1]
    n("Mul", ["iota_r", "rp"], ["rmax_src"])          # iota where present else 0
    n("ReduceMax", ["rmax_src"], ["y1"], axes=[2], keepdims=1)   # [1,1,1,1]
    # cols
    n("Mul", ["iota_c", "cp"], ["icp"])
    n("Sub", ["one", "cp"], ["inv_cp"])
    n("Mul", ["inv_cp", "big"], ["bigcp"])
    n("Add", ["icp", "bigcp"], ["cmin_src"])
    n("ReduceMin", ["cmin_src"], ["x0"], axes=[3], keepdims=1)
    n("Mul", ["iota_c", "cp"], ["cmax_src"])
    n("ReduceMax", ["cmax_src"], ["x1"], axes=[3], keepdims=1)
    # sy=y0+y1, sx=x0+x1, off=y0-x0  (all [1,1,1,1])
    n("Add", ["y0", "y1"], ["sy"])
    n("Add", ["x0", "x1"], ["sx"])
    n("Sub", ["y0", "x0"], ["off"])
    # Build 30x30 permutation matrices. outer[i,r]=A2d[i]+B2d[r]
    n("Add", ["A2d", "B2d"], ["sumIR"])     # [30,30] i+r
    n("Sub", ["A2d", "B2d"], ["difIR"])     # [30,30] i-r
    # Pr = (sumIR == sy) ; need sy as [1,1] scalar broadcastable to [30,30].
    # sy is [1,1,1,1]; reshape to scalar [] for Equal broadcast with [30,30].
    inits.append(c("shp_s", np.array([], np.int64)))
    n("Reshape", ["sy", "shp_s"], ["sy_s"])
    n("Reshape", ["sx", "shp_s"], ["sx_s"])
    n("Reshape", ["off", "shp_s"], ["off_s"])
    n("Equal", ["sumIR", "sy_s"], ["PrB"]); n("Cast", ["PrB"], ["Pr"], to=TensorProto.FLOAT)
    n("Equal", ["sumIR", "sx_s"], ["PcB"]); n("Cast", ["PcB"], ["Pc"], to=TensorProto.FLOAT)
    n("Equal", ["difIR", "off_s"], ["SrB"]); n("Cast", ["SrB"], ["Sr"], to=TensorProto.FLOAT)
    # transpose of input over (H,W)
    n("Transpose", ["input"], ["Xt"], perm=[0, 1, 3, 2])
    # Tg = Sr @ Xt @ Sr   (Sr is [30,30], broadcasts over batch/chan)
    n("MatMul", ["Sr", "Xt"], ["SrXt"])
    n("MatMul", ["SrXt", "Sr"], ["Tg"])
    # The 8 transforms
    n("MatMul", ["Pr", "input"], ["udX"])         # Pr@X
    n("MatMul", ["input", "Pc"], ["lrX"])         # X@Pc
    n("MatMul", ["Pr", "lrX"], ["rotX"])          # Pr@X@Pc
    n("MatMul", ["Pr", "Tg"], ["udT"])
    n("MatMul", ["Tg", "Pc"], ["lrT"])
    n("MatMul", ["Pr", "lrT"], ["rotT"])
    # union
    n("Max", ["input", "udX", "lrX", "rotX", "Tg", "udT", "lrT", "rotT"], ["output"])

    inits = [it for it in inits]  # already appended dynamically above
    g = helper.make_graph(
        N, "task020",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("saved", path, len(m.SerializeToString()), "bytes")


if __name__ == "__main__":
    build()
