"""Build task037.onnx — compact diagonal segment-fill (bool/int8 working tensors).

Rule: each non-background color appears as exactly two cells lying on a perfect
diagonal (|dr|==|dc|). Output draws the straight diagonal segment connecting the
two endpoints (inclusive). Background fills everything else.

Algorithm (no labeling, pure shift + bool Or/And):
  Work on the 10x10 active region, 10 one-hot channels, cast to BOOL (1 byte).
  For each colored channel, segment = cells reachable from an endpoint going one
  diagonal direction AND reachable going the opposite direction. Endpoints lie on
  a single diagonal so:
      seg = (cumOR(+1,+1) & cumOR(-1,-1)) | (cumOR(+1,-1) & cumOR(-1,+1))
  cumOR along a direction uses doubling shifts 1,2,4,8 (covers any 10x10 diagonal).
  Shifts via Pad(leading)+Slice(trailing). Or == max, And == min on bool.
  Background channel 0 is excluded from propagation, recomputed at the end as
  NOT(any colored channel), so each cell decodes to exactly one color.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf2_task037.onnx"
R = 10  # active region side


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    inits = [
        # slice channels 1..9 and the 10x10 active region in one Slice
        const("starts", np.array([1, 0, 0], np.int64)),
        const("ends", np.array([10, R, R], np.int64)),
        const("axes123", np.array([1, 2, 3], np.int64)),
        const("ax1", np.array([1], np.int64)),
        const("zero8", np.array(0, np.int8)),
        # pad full [1,10,10,10] back to [1,10,30,30]
        const("pad30", np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # slice [1,10,30,30] -> [1,9,10,10] (colored channels, active region), cast bool
    n("Slice", ["input", "starts", "ends", "axes123"], ["col0"])
    n("Cast", ["col0"], ["col"], to=TensorProto.BOOL)   # [1,9,10,10] bool

    def cum(src, dr, dc, tag):
        # cumulative-OR along (dr,dc) via doubling shifts; each shift is a single
        # shape-preserving Pad with positive lead / negative trailing pads.
        cur = src
        for s in (1, 2, 4):
            ddr, ddc = dr * s, dc * s
            # translate content by (+ddr,+ddc): leading pad = d (negative crops),
            # trailing pad = -d. Shape preserved; zeros shifted in.
            pads = np.array([0, 0, ddr, ddc, 0, 0, -ddr, -ddc], np.int64)
            inits.append(const(f"pads_{tag}_{s}", pads))
            n("Pad", [cur, f"pads_{tag}_{s}"], [f"shift_{tag}_{s}"])
            n("Or", [cur, f"shift_{tag}_{s}"], [f"cum_{tag}_{s}"])
            cur = f"cum_{tag}_{s}"
        return cur

    fmm = cum("col", 1, 1, "mm")
    bmm = cum("col", -1, -1, "bm")
    fma = cum("col", 1, -1, "ma")
    bma = cum("col", -1, 1, "ba")

    n("And", [fmm, bmm], ["seg_main"])
    n("And", [fma, bma], ["seg_anti"])
    n("Or", ["seg_main", "seg_anti"], ["seg"])   # [1,9,10,10] bool colored channels

    # background channel (bool) = NOT(any colored channel)
    n("Cast", ["seg"], ["segc"], to=TensorProto.INT8)          # [1,9,10,10] int8
    n("ReduceMax", ["segc", "ax1"], ["anymax"], keepdims=1)    # [1,1,10,10] int8
    n("Equal", ["anymax", "zero8"], ["bgb"])                   # bool: True where no color
    n("Concat", ["bgb", "seg"], ["fullb"], axis=1)             # [1,10,10,10] bool
    n("Cast", ["fullb"], ["filled"], to=TensorProto.FLOAT)     # [1,10,10,10] float

    # pad back to [1,10,30,30] with zeros (matches expected zero padding region)
    n("Pad", ["filled", "pad30"], ["output"])

    graph = helper.make_graph(
        N, "task037",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT, len(model.SerializeToString()), "bytes file")


if __name__ == "__main__":
    build()
