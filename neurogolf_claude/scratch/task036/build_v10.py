"""task036 v10: channel-space density discriminator + dynamic Slice/Pad crop.
Reshape mask to [30,30] so row/col reductions yield [30] vectors and ArgMax /
ReduceSum give [1] directly, removing the start/size Reshape nodes.
File-size scored (data-dependent crop).
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/b4_task036.onnx"
F = TensorProto.FLOAT
I64 = TensorProto.INT64


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    pen = np.zeros((1, 10, 1, 1), np.float32); pen[0, 0] = 100.0
    inits = [
        c("pen", pen),
        c("sa", np.array([2, 3], np.int64)),
        c("zf", np.float32(0.0)),
        c("z6", np.array([0, 0, 0, 0, 0, 0], np.int64)),
        c("g2", np.array([30, 30], np.int64)),
        c("a0", np.array([0], np.int64)),
        c("T", np.array([30, 30], np.int64)),
    ]

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    n("AveragePool", ["input"], ["dens"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      count_include_pad=1)
    n("ReduceMax", ["dens"], ["score"], axes=[2, 3], keepdims=1)
    n("Sub", ["score", "pen"], ["sadj"])
    n("ReduceMax", ["sadj"], ["smax"], axes=[1], keepdims=1)
    n("Equal", ["sadj", "smax"], ["selb"])
    n("Cast", ["selb"], ["sel"], to=F)
    n("Conv", ["input", "sel"], ["Mf"])                    # [1,1,30,30]
    n("Cast", ["Mf"], ["m4"], to=I64)
    n("Reshape", ["m4", "g2"], ["M"])                      # [30,30]
    n("ReduceMax", ["M"], ["kr"], axes=[1], keepdims=0)    # [30] per-row any
    n("ReduceMax", ["M"], ["kc"], axes=[0], keepdims=0)    # [30] per-col any
    n("ArgMax", ["kr"], ["rmn"], axis=0, keepdims=1)       # [1]
    n("ArgMax", ["kc"], ["cmn"], axis=0, keepdims=1)
    n("ReduceSum", ["kr", "a0"], ["h"], keepdims=1)        # [1]
    n("ReduceSum", ["kc", "a0"], ["w"], keepdims=1)
    n("Concat", ["rmn", "cmn"], ["st"], axis=0)            # [2]
    n("Concat", ["h", "w"], ["hw"], axis=0)                # [2]
    n("Add", ["st", "hw"], ["en"])                         # [2]
    n("Slice", ["input", "st", "en", "sa"], ["crop"])
    n("Sub", ["T", "hw"], ["pa"])
    n("Concat", ["z6", "pa"], ["po"], axis=0)
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
    print("saved", OUT, len(model.SerializeToString()), "bytes", "nodes", len(N))


if __name__ == "__main__":
    build()
