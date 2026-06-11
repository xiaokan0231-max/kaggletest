"""Build task014.onnx.

Rule: grid is tiled with noisy blocks of one common color; exactly one block
uses a unique color. Output = bounding-box crop of the unique color.

Pipeline:
  1. count per channel -> pick the rarest nonzero color present (one-hot sel).
  2. M = Conv(input, sel)  (dynamic 1x1 conv weight) -> target mask [1,1,30,30].
  3. keep_r / keep_c indicators -> CumSum -> permutation matrices Sr, T.
  4. M2 = Sr @ M @ T (row/col compaction to top-left).
  5. G_out = u * v from row/col sums of Sr / T.
  6. output = Conv(Concat(G,M2), W) with W built from e0 and sel-e0.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions/task014.onnx"
BIG = 4096.0


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    inits = [
        const("bigch", (BIG * np.array([0] + [1] * 9, np.float32)).reshape(1, 10, 1, 1)),
        const("bigs", np.float32(BIG)),
        const("one", np.float32(1.0)),
        const("iota_col", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
        const("iota_row", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
        const("e0", np.eye(10, dtype=np.float32)[0].reshape(1, 10, 1, 1)),
        const("ax23", np.array([2, 3], np.int64)),
        const("ax3", np.array([3], np.int64)),
        const("ax2", np.array([2], np.int64)),
        const("cax3", np.array(3, np.int64)),
        const("sh_row", np.array([1, 1, 1, 30], np.int64)),
        const("sh_col", np.array([1, 1, 30, 1], np.int64)),
        const("sh_w", np.array([10, 1, 1, 1], np.int64)),
    ]
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # 1) rarest nonzero color present
    n("ReduceSum", ["input", "ax23"], ["count"], keepdims=1)        # [1,10,1,1]
    n("Sign", ["count"], ["present"])
    n("Mul", ["bigch", "present"], ["pen"])
    n("Add", ["count", "bigs"], ["cb"])
    n("Sub", ["cb", "pen"], ["score"])                               # count if eligible else count+BIG
    n("Neg", ["score"], ["nscore"])
    n("ReduceMax", ["nscore"], ["nmin"], axes=[1], keepdims=1)
    n("Neg", ["nmin"], ["minv"])
    n("Sub", ["score", "minv"], ["sdiff"])
    n("Sign", ["sdiff"], ["ssign"])
    n("Sub", ["one", "ssign"], ["sel"])                              # [1,10,1,1] one-hot

    # 2) target mask via dynamic 1x1 conv
    n("Conv", ["input", "sel"], ["M"])                               # [1,1,30,30]

    # 3) row permutation Sr[i,r]
    n("ReduceMax", ["M"], ["krA"], axes=[3], keepdims=1)             # [1,1,30,1]
    n("Reshape", ["krA", "sh_row"], ["kr"])                          # [1,1,1,30]
    n("CumSum", ["kr", "cax3"], ["pr0"])
    n("Sub", ["pr0", "one"], ["pr"])
    n("Equal", ["iota_col", "pr"], ["ErB"])                          # [1,1,30,30]
    n("Cast", ["ErB"], ["Er"], to=TensorProto.FLOAT)
    n("Mul", ["Er", "kr"], ["Sr"])

    #    column permutation T[c,j]
    n("ReduceMax", ["M"], ["kcA"], axes=[2], keepdims=1)             # [1,1,1,30]
    n("CumSum", ["kcA", "cax3"], ["pc0"])
    n("Sub", ["pc0", "one"], ["pcr"])
    n("Reshape", ["pcr", "sh_col"], ["pc"])                          # [1,1,30,1]
    n("Reshape", ["kcA", "sh_col"], ["kc"])                          # [1,1,30,1]
    n("Equal", ["pc", "iota_row"], ["EcB"])
    n("Cast", ["EcB"], ["Ec"], to=TensorProto.FLOAT)
    n("Mul", ["Ec", "kc"], ["T"])

    # 4) compaction
    n("MatMul", ["Sr", "M"], ["A"])
    n("MatMul", ["A", "T"], ["M2"])                                  # [1,1,30,30]

    # 5) output-grid mask
    n("ReduceSum", ["Sr", "ax3"], ["u"], keepdims=1)                 # [1,1,30,1]
    n("ReduceSum", ["T", "ax2"], ["v"], keepdims=1)                  # [1,1,1,30]
    n("Mul", ["u", "v"], ["G"])                                      # [1,1,30,30]

    # 6) assemble via dynamic 1x1 conv: out_c = e0_c*G + (sel_c-e0_c)*M2
    n("Concat", ["G", "M2"], ["X2"], axis=1)                         # [1,2,30,30]
    n("Sub", ["sel", "e0"], ["coef"])
    n("Reshape", ["e0", "sh_w"], ["w0"])
    n("Reshape", ["coef", "sh_w"], ["w1"])
    n("Concat", ["w0", "w1"], ["W"], axis=1)                         # [10,2,1,1]
    n("Conv", ["X2", "W"], ["output"])                               # [1,10,30,30]

    graph = helper.make_graph(
        N, "task014",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT, len(model.SerializeToString()), "bytes file")


if __name__ == "__main__":
    build()
