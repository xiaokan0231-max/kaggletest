"""task036 v8: channel-space density discriminator (1 AvgPool, like deployed) +
single-plane color recovery + dynamic Slice/Pad crop. Aims for fewest nodes so
the (file-size-scored) model is smallest.
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
        c("one", np.float32(1.0)),
        c("sa", np.array([2, 3], np.int64)),
        c("zf", np.float32(0.0)),
        c("z4", np.array([0, 0, 0, 0], np.int64)),
        c("z2", np.array([0, 0], np.int64)),
        c("d1", np.array([1], np.int64)),
        c("ax2", np.array([2], np.int64)),
        c("ax3", np.array([3], np.int64)),
        c("T30", np.int64(30)),
    ]

    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))

    # density argmax -> one-hot channel selector sel [1,10,1,1]
    n("AveragePool", ["input"], ["dens"], kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      strides=[1, 1], count_include_pad=1)
    n("ReduceMax", ["dens"], ["score"], axes=[2, 3], keepdims=1)
    n("Sub", ["score", "pen"], ["sadj"])
    n("ReduceMax", ["sadj"], ["smax"], axes=[1], keepdims=1)
    n("Sub", ["sadj", "smax"], ["sd"])
    n("Sign", ["sd"], ["ss"])
    n("Add", ["ss", "one"], ["sel"])                    # [1,10,1,1] one-hot
    # shape mask M [1,1,30,30] = Conv(input, sel)
    n("Conv", ["input", "sel"], ["M"])
    # contiguous bbox from M
    n("Cast", ["M"], ["mi"], to=I64)
    n("ReduceMax", ["mi"], ["kr"], axes=[3], keepdims=1)
    n("ReduceMax", ["mi"], ["kc"], axes=[2], keepdims=1)
    n("ArgMax", ["kr"], ["rmn"], axis=2, keepdims=1)
    n("ReduceSum", ["kr", "ax2"], ["h"], keepdims=1)
    n("ArgMax", ["kc"], ["cmn"], axis=3, keepdims=1)
    n("ReduceSum", ["kc", "ax3"], ["w"], keepdims=1)
    n("Reshape", ["rmn", "d1"], ["rmn1"])
    n("Reshape", ["cmn", "d1"], ["cmn1"])
    n("Concat", ["rmn1", "cmn1"], ["st"], axis=0)
    n("Add", ["rmn", "h"], ["re"])
    n("Add", ["cmn", "w"], ["ce"])
    n("Reshape", ["re", "d1"], ["re1"])
    n("Reshape", ["ce", "d1"], ["ce1"])
    n("Concat", ["re1", "ce1"], ["en"], axis=0)
    n("Slice", ["input", "st", "en", "sa"], ["crop"])
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
    print("saved", OUT, len(model.SerializeToString()), "bytes", "nodes", len(N))


if __name__ == "__main__":
    build()
