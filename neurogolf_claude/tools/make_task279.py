"""Build task279.onnx — compact enclosed-shape recolor (pure bool flood-fill).

Rule: foreground (color 1) shapes that ENCLOSE at least one background hole (a
background-9 region not connected to the grid border) are recolored to 8; shapes
with no enclosed hole stay 1. Background (9) and the holes themselves unchanged.

Algorithm (pure bool flood-fill, no connected-component labeling, no Loop/Scan):
  Work in a [1,1,A,A] single-channel region (A=16 == max grid; the region border
  itself is the outside-source, so no padding ring is needed).
  fg = input ch1, bg = input ch9, ingrid = ReduceMax over channels (1 in grid).
  open   = bg OR (NOT ingrid)             cells "light" can travel through
  reach  = flood from the AxA border ring through `open` (4-conn, Kr iters):
           reach <- (dil4(reach)) AND open
  enclosed = bg AND NOT reach             bg cells unreachable from outside
  seed0  = fg AND dil4(enclosed)          fg cells touching a hole
  recolor= flood seed0 across fg (4-conn, Ks iters): seed <- dil4(seed) AND fg
  Output one-hot: ch9=bg, ch1=fg AND NOT recolor, ch8=recolor, others copied.

All working tensors BOOL (1 byte) in a 16x16 region => 256 bytes each.
Or==Max, And==Min, NOT==Equal(x,0). 4-conn dilate = 4 Pad-shifts OR'd with self.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = "/tmp/wf3_task279.onnx"
A = 16          # working region side (== max grid; border ring is the outside)
K_REACH = 16    # reach flood iterations (4-conn). min observed 13, margin to 16.
K_SEED = 13     # seed flood iterations (4-conn). min observed 8, margin to 13.


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    inits = []
    N = []

    def n(op, ins, outs, **kw):
        N.append(helper.make_node(op, ins, outs, **kw))

    inits += [
        const("st_fg", np.array([1, 0, 0], np.int64)),
        const("en_fg", np.array([2, A, A], np.int64)),
        const("st_bg", np.array([9, 0, 0], np.int64)),
        const("en_bg", np.array([10, A, A], np.int64)),
        const("st_all", np.array([0, 0, 0], np.int64)),
        const("en_all", np.array([10, A, A], np.int64)),
        const("ax123", np.array([1, 2, 3], np.int64)),
        const("ax1", np.array([1], np.int64)),
        const("falsei8", np.array(0, np.int8)),
    ]
    # fg, bg as bool [1,1,A,A]
    n("Slice", ["input", "st_fg", "en_fg", "ax123"], ["fg_f"])
    n("Cast", ["fg_f"], ["fg"], to=TensorProto.BOOL)
    n("Slice", ["input", "st_bg", "en_bg", "ax123"], ["bg_f"])
    n("Cast", ["bg_f"], ["bg"], to=TensorProto.BOOL)
    # ingrid = any channel present
    n("Slice", ["input", "st_all", "en_all", "ax123"], ["allc"])
    n("ReduceMax", ["allc", "ax1"], ["ingrid_f"], keepdims=1)
    n("Cast", ["ingrid_f"], ["ingrid"], to=TensorProto.BOOL)
    n("Cast", ["ingrid"], ["ingrid_i8"], to=TensorProto.INT8)
    n("Equal", ["ingrid_i8", "falsei8"], ["pad"])          # NOT ingrid
    n("Or", ["bg", "pad"], ["open"])

    # border ring seed
    ring = np.zeros((1, 1, A, A), np.bool_)
    ring[:, :, 0, :] = True
    ring[:, :, -1, :] = True
    ring[:, :, :, 0] = True
    ring[:, :, :, -1] = True
    inits.append(const("ring", ring))
    n("And", ["open", "ring"], ["reach0"])

    # 4-conn dilation: OR self with 4 single-cell Pad-shifts
    def dilate4(src, tag):
        cur = src
        for k, (dr, dc) in {"u": (-1, 0), "d": (1, 0),
                            "l": (0, -1), "r": (0, 1)}.items():
            pname = f"p_{tag}_{k}"
            inits.append(const(pname, np.array(
                [0, 0, dr, dc, 0, 0, -dr, -dc], np.int64)))
            n("Pad", [src, pname], [f"{tag}_{k}"])
            nxt = f"{tag}_o{k}"
            n("Or", [cur, f"{tag}_{k}"], [nxt])
            cur = nxt
        return cur

    # reach flood
    cur = "reach0"
    for i in range(K_REACH):
        dn = dilate4(cur, f"rd{i}")
        nxt = f"reach{i+1}"
        n("And", [dn, "open"], [nxt])
        cur = nxt
    reach = cur
    n("Cast", [reach], ["reach_i8"], to=TensorProto.INT8)
    n("Equal", ["reach_i8", "falsei8"], ["notreach"])
    n("And", ["bg", "notreach"], ["enclosed"])

    # seed0 = fg AND dilate4(enclosed)
    encd = dilate4("enclosed", "encd")
    n("And", ["fg", encd], ["seed0"])
    cur = "seed0"
    for i in range(K_SEED):
        dn = dilate4(cur, f"sd{i}")
        nxt = f"seed{i+1}"
        n("And", [dn, "fg"], [nxt])
        cur = nxt
    recolor = cur

    # assemble output channels
    n("Cast", [recolor], ["recolor_i8"], to=TensorProto.INT8)
    n("Equal", ["recolor_i8", "falsei8"], ["notrecolor"])
    n("And", ["fg", "notrecolor"], ["stay1"])

    n("Cast", ["bg"], ["bg_o"], to=TensorProto.FLOAT)
    n("Cast", ["stay1"], ["s1_o"], to=TensorProto.FLOAT)
    n("Cast", [recolor], ["rc_o"], to=TensorProto.FLOAT)

    keep_chs = [0, 2, 3, 4, 5, 6, 7]
    parts = {}
    for ch in keep_chs:
        inits.append(const(f"cs_{ch}", np.array([ch, 0, 0], np.int64)))
        inits.append(const(f"ce_{ch}", np.array([ch + 1, A, A], np.int64)))
        n("Slice", ["input", f"cs_{ch}", f"ce_{ch}", "ax123"], [f"ch{ch}"])
        parts[ch] = f"ch{ch}"
    parts[1] = "s1_o"
    parts[8] = "rc_o"
    parts[9] = "bg_o"
    ordered = [parts[c] for c in range(10)]
    n("Concat", ordered, ["full_AA"], axis=1)

    inits.append(const("pad30", np.array(
        [0, 0, 0, 0, 0, 0, 30 - A, 30 - A], np.int64)))
    n("Pad", ["full_AA", "pad30"], ["output"])

    graph = helper.make_graph(
        N, "task279",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print("saved", OUT, len(model.SerializeToString()), "file bytes; nodes", len(N))


if __name__ == "__main__":
    build()
