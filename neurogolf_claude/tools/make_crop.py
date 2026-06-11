"""Flexible positional crop builder (rows/cols spec) on top of cropkit.

Row/col spec (a string):
  all              -> keep every present row/col
  first:K          -> first K  (K integer constant)
  last:K           -> last K
  firstfrac:F      -> first floor(D*F)  where D = grid height(rows)/width(cols)
  lastfrac:F       -> last  floor(D*F)
  firstotherfrac:F -> first floor(Dother*F)  (rows use width, cols use height)

Usage: python make_crop.py task067 --rows all --cols firstfrac:0.3333333
       python make_crop.py task135 --rows first:3 --cols last:3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import cropkit as ck  # noqa: E402
import numpy as np  # noqa: E402
from onnx import helper  # noqa: E402

SOL = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/solutions"


def emit_dims(n, inits):
    """presence, row_present[1,1,30,1], col_present[1,1,1,30], H[1,1,1,1], W[1,1,1,1],
    plus nonzero band masks nz_row/nz_col (cells with a nonzero color)."""
    inits.append(ck.const("mc_ax2", np.array([2], np.int64)))
    inits.append(ck.const("mc_ax3", np.array([3], np.int64)))
    inits.append(ck.const("mc_wnz", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1)))
    n("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)        # [1,1,30,30]
    n("ReduceMax", ["presence"], ["row_present"], axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["presence"], ["col_present"], axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["row_present", "mc_ax2"], ["Hdim"], keepdims=1)      # [1,1,1,1]
    n("ReduceSum", ["col_present", "mc_ax3"], ["Wdim"], keepdims=1)      # [1,1,1,1]
    n("Conv", ["input", "mc_wnz"], ["nz"])                               # [1,1,30,30] nonzero indicator
    n("ReduceMax", ["nz"], ["nz_row"], axes=[3], keepdims=1)             # [1,1,30,1]
    n("ReduceMax", ["nz"], ["nz_col"], axes=[2], keepdims=1)             # [1,1,1,30]


def emit_axis(n, inits, which, spec, out):
    """which='row' (axis2, iota=ck_iota_r, present=row_present, dim=Hdim, other=Wdim)
            ='col' (axis3, iota=ck_iota_c, present=col_present, dim=Wdim, other=Hdim)."""
    if which == "row":
        iota, present, dim, other, nzb = "ck_iota_r", "row_present", "Hdim", "Wdim", "nz_row"
    else:
        iota, present, dim, other, nzb = "ck_iota_c", "col_present", "Wdim", "Hdim", "nz_col"
    p = out
    if spec == "all":
        n("Identity", [present], [out])
        return
    # band-rank selection: (nz)?half / (nz)?first:K / (nz)?last:K on present or nonzero band
    if spec in ("firsthalf", "lasthalf", "nzfirsthalf", "nzlasthalf") or \
       spec.startswith(("first:", "last:", "nzfirst:", "nzlast:")):
        band = nzb if spec.startswith("nz") else present
        s = spec[2:] if spec.startswith("nz") else spec
        inits.append(ck.const(f"{p}_one", np.float32(1.0)))
        inits.append(ck.const(f"{p}_axc", np.array(2 if which == "row" else 3, np.int64)))
        inits.append(ck.const(f"{p}_axr", np.array([2] if which == "row" else [3], np.int64)))
        # reshape band to a vector along its axis for cumsum
        n("CumSum", [band, f"{p}_axc"], [f"{p}_cs"])
        n("Sub", [f"{p}_cs", f"{p}_one"], [f"{p}_rank"])              # rank within band
        n("ReduceSum", [band, f"{p}_axr"], [f"{p}_B"], keepdims=1)    # band size
        if s in ("firsthalf", "lasthalf"):
            inits.append(ck.const(f"{p}_half", np.float32(0.5)))
            n("Mul", [f"{p}_B", f"{p}_half"], [f"{p}_kf"]); n("Floor", [f"{p}_kf"], [f"{p}_K"])
            kind = "first" if s == "firsthalf" else "last"
        else:
            kind, kv = s.split(":")
            inits.append(ck.const(f"{p}_K", np.float32(float(kv))))
        if kind == "first":
            n("Less", [f"{p}_rank", f"{p}_K"], [f"{p}_m"])
        else:
            n("Sub", [f"{p}_B", f"{p}_K"], [f"{p}_lo"])
            n("GreaterOrEqual", [f"{p}_rank", f"{p}_lo"], [f"{p}_m"])
        n("Cast", [f"{p}_m"], [f"{p}_mf"], to=1)
        n("Mul", [f"{p}_mf", band], [out])
        return
    if spec == "nzall":
        n("Identity", [nzb], [out])
        return
    kind, val = spec.split(":")
    if kind in ("first", "last"):
        k_name = f"{p}_k"
        inits.append(ck.const(k_name, np.float32(float(val))))
    else:  # frac variants
        f_name = f"{p}_f"
        inits.append(ck.const(f_name, np.float32(float(val))))
        base = dim if kind in ("firstfrac", "lastfrac") else other
        n("Mul", [base, f_name], [f"{p}_kf"])
        n("Floor", [f"{p}_kf"], [f"{p}_k"])
        k_name = f"{p}_k"
        kind = "first" if kind.startswith("first") else "last"
    if kind == "first":
        n("Less", [iota, k_name], [f"{p}_lt"])
        n("Cast", [f"{p}_lt"], [f"{p}_ltf"], to=1)  # FLOAT
        n("Mul", [f"{p}_ltf", present], [out])
    else:  # last K: iota >= dim - K
        n("Sub", [dim, k_name], [f"{p}_lo"])
        n("GreaterOrEqual", [iota, f"{p}_lo"], [f"{p}_ge"])
        n("Cast", [f"{p}_ge"], [f"{p}_gef"], to=1)
        n("Mul", [f"{p}_gef", present], [out])


def build(tid, rows, cols):
    inits = ck.base_inits()
    N = []
    n = lambda op, ins, outs, **kw: N.append(helper.make_node(op, ins, outs, **kw))
    emit_dims(n, inits)
    emit_axis(n, inits, "row", rows, "keeprow")
    emit_axis(n, inits, "col", cols, "keepcol")
    ck.emit_compaction(n, "keeprow", "keepcol", "input", "output")
    ck.finalize(N, inits, tid, f"{SOL}/{tid}.onnx")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tid")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cols", required=True)
    a = ap.parse_args()
    build(a.tid, a.rows, a.cols)
