"""task036 v9: channel-space density discriminator + dynamic Slice/Pad crop,
with batched [2]-vector bbox math to cut Reshape nodes. File-size scored.
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
        c("d2", np.array([2], np.int64)),
        c("ax2", np.array([2], np.int64)),
        c("ax3", np.array([3], np.int64)),
        c("T", np.array([30, 30], np.int64)),
    ]

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # density argmax -> one-hot channel selector
    n("AveragePool", ["input"], ["dens"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      count_include_pad=1)
    n("ReduceMax", ["dens"], ["score"], axes=[2, 3], keepdims=1)
    n("Sub", ["score", "pen"], ["sadj"])
    n("ReduceMax", ["sadj"], ["smax"], axes=[1], keepdims=1)
    n("Equal", ["sadj", "smax"], ["selb"])                 # 1 at argmax channel
    n("Cast", ["selb"], ["sel"], to=F)
    n("Conv", ["input", "sel"], ["M"])                     # [1,1,30,30] shape mask
    n("Cast", ["M"], ["mi"], to=I64)
    n("ReduceMax", ["mi"], ["kr"], axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["mi"], ["kc"], axes=[2], keepdims=1)   # [1,1,1,30]
    # mins via ArgMax (first kept index); sizes via ReduceSum
    n("ArgMax", ["kr"], ["rmn"], axis=2, keepdims=1)       # [1,1,1,1]
    n("ArgMax", ["kc"], ["cmn"], axis=3, keepdims=1)
    n("ReduceSum", ["kr", "ax2"], ["h"], keepdims=1)
    n("ReduceSum", ["kc", "ax3"], ["w"], keepdims=1)
    # starts [2] = [rmn, cmn]
    n("Concat", ["rmn", "cmn"], ["st4"], axis=0)           # [2,1,1,1]
    n("Reshape", ["st4", "d2"], ["st"])                    # [2]
    # sizes [2] = [h, w]
    n("Concat", ["h", "w"], ["hw4"], axis=0)
    n("Reshape", ["hw4", "d2"], ["hw"])                    # [2]
    n("Add", ["st", "hw"], ["en"])                         # ends [2]
    n("Slice", ["input", "st", "en", "sa"], ["crop"])
    # pad-after [2] = [30,30] - [h,w]
    n("Sub", ["T", "hw"], ["pa"])                          # [2]
    n("Concat", ["z6", "pa"], ["po"], axis=0)              # [8] = begins(4)+[0,0]+[h1,w1]
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
