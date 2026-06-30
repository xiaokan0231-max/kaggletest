"""Build compact ONNX for task230.

Rule (verified 262/262): every isolated 2x2 block of color 5 gets four corner
markers placed one cell diagonally outside the block:
  1 at top-left-diag (r-1,c-1), 2 at top-right-diag (r-1,c+2),
  3 at bottom-left-diag (r+2,c-1), 4 at bottom-right-diag (r+2,c+2),
where (r,c) is the block's top-left cell. Markers only land on background.
No corner ever falls outside the grid in any of the 262 cases, so a full-30x30
model is exact (blocks are always >=1 cell from the edge in the marker direction).

Construction (all interior tensors uint8/bool, [1,1,30,30] = 900B):
  g  = Conv(input, [0..9])                       # f32 color grid (input-collapse)
  is5 = Equal(g,5)                               # bool [1,1,30,30]
  block presence B[r,c]=1 iff 2x2 all-5 with top-left (r,c):
      B = is5 & shift_left(is5) & shift_up(is5) & shift_upleft(is5)
  marker maps are shifts of B:
      M1 (val1) at (r-1,c-1) = shift B up+left by 1
      M2 (val2) at (r-1,c+2) = shift B up by1, right by2
      M3 (val3) at (r+2,c-1) = shift B down by2, left by1
      M4 (val4) at (r+2,c+2) = shift B down by2, right by2
  gm = g_u8 + 1*M1 + 2*M2 + 3*M3 + 4*M4          # uint8 color grid (markers on bg)
  output = Equal(gm[1,1,30,30], colors[1,10,1,1] uint8 0..9) -> bool [1,10,30,30]
The only [1,10,*] tensor is the excluded graph output.

Shifts are done with Pad (pad on one side, the Slice-equivalent crop happens
because Pad+Slice). We use Slice for cropping then Pad to restore size, or simply
Pad with negative? ONNX Pad doesn't crop. So a shift = Slice(drop one edge) then
Pad(add zeros on opposite edge). To keep op count low we instead realise each
marker map directly as a single Conv on B with a 3x3 (or larger) kernel that has a
1 at the source offset, using pads to keep [1,1,30,30]. A shift by (dr,dc) is a
correlation with a kernel that has 1 at position (-dr,-dc) relative to centre.
We need offsets up to +-2, so a 5x5 kernel (centre at (2,2)) with pads=2 covers
all four shifts; we can pack all four markers (weighted 1,2,3,4) into ONE 5x5
conv producing the combined marker grid directly.
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def build(path):
    nodes, inits = [], []

    # 1) color grid from full f32 one-hot input (input excluded) -> g [1,1,30,30] f32
    #    AND a validity mask: padding cells (beyond the actual grid) have an all-zero
    #    one-hot, in-grid cells sum to 1. Pack both into one Conv with 2 output filters:
    #    filter0 = [0..9] -> color grid; filter1 = ones -> per-cell channel sum (validity).
    Wc = np.zeros((2, 10, 1, 1), np.float32)
    Wc[0, :, 0, 0] = np.arange(10, dtype=np.float32)  # color
    Wc[1, :, 0, 0] = 1.0                              # validity (sum of one-hot)
    inits.append(cf("Wcol", Wc))
    nodes.append(helper.make_node("Conv", ["input", "Wcol"], ["gv"]))  # f32 [1,2,30,30]
    inits += [ci("s0", [0]), ci("s1", [1]), ci("s2", [2]), ci("ax1", [1])]
    nodes.append(helper.make_node("Slice", ["gv", "s0", "s1", "ax1"], ["g"]))      # [1,1,30,30] color
    nodes.append(helper.make_node("Slice", ["gv", "s1", "s2", "ax1"], ["validf"]))  # [1,1,30,30] sum
    inits.append(cf("half", 0.5))
    nodes.append(helper.make_node("Greater", ["validf", "half"], ["valid"]))  # bool in-grid mask

    # 2) is5 mask as float (for conv-based block detection)
    inits.append(cf("five", 5.0))
    nodes.append(helper.make_node("Equal", ["g", "five"], ["is5b"]))   # bool
    nodes.append(helper.make_node("Cast", ["is5b"], ["is5"], to=F32))  # f32 0/1 [1,1,30,30]

    # 3) block presence B[r,c] = 1 iff 2x2 window (r,c),(r,c+1),(r+1,c),(r+1,c+1) all 5.
    #    Correlation of is5 with a 2x2 kernel of ones gives the count over the window
    #    whose TOP-LEFT is the output position when we pad on the bottom/right.
    #    Conv (cross-correlation) with kernel K[ki,kj] sums is5[r+ki, c+kj].
    #    With kernel ones[2,2] and pads (top=0,left=0,bottom=1,right=1) the output at
    #    (r,c) = sum of is5 over rows r..r+1, cols c..c+1 -> ==4 iff block. [1,1,30,30].
    inits.append(cf("Wblk", np.ones((1, 1, 2, 2), np.float32)))
    nodes.append(helper.make_node("Conv", ["is5", "Wblk"], ["bcount"],
                                  pads=[0, 0, 1, 1]))  # f32 [1,1,30,30], pad bottom/right
    inits.append(cf("three5", 3.5))
    nodes.append(helper.make_node("Greater", ["bcount", "three5"], ["Bb"]))  # bool, ==4>3.5
    nodes.append(helper.make_node("Cast", ["Bb"], ["B"], to=F32))            # f32 0/1 [1,1,30,30]

    # 4) combined marker grid via ONE 5x5 conv on B.
    #    output[r,c] = sum_{ki,kj} W[ki,kj] * B[r + ki-2, c + kj-2]   (Conv corr, pad=2)
    #    We want marker value v deposited at block-corner offsets:
    #      M1 val1 at (r-1,c-1) from block at (r,c): out cell (R,C) gets 1 if B[R+1,C+1]=1
    #         -> need ki-2=+1 -> ki=3 ; kj-2=+1 -> kj=3 ; W[3,3]=1
    #      M2 val2 at (r-1,c+2): out (R,C) gets 2 if B[R+1, C-2]=1 -> ki=3, kj=0 ; W[3,0]=2
    #      M3 val3 at (r+2,c-1): out (R,C) gets 3 if B[R-2, C+1]=1 -> ki=0, kj=3 ; W[0,3]=3
    #      M4 val4 at (r+2,c+2): out (R,C) gets 4 if B[R-2, C-2]=1 -> ki=0, kj=0 ; W[0,0]=4
    W5 = np.zeros((1, 1, 5, 5), np.float32)
    W5[0, 0, 3, 3] = 1.0
    W5[0, 0, 3, 0] = 2.0
    W5[0, 0, 0, 3] = 3.0
    W5[0, 0, 0, 0] = 4.0
    inits.append(cf("Wmrk", W5))
    nodes.append(helper.make_node("Conv", ["B", "Wmrk"], ["markf"],
                                  pads=[2, 2, 2, 2]))  # f32 [1,1,30,30], values in {0,1,2,3,4}

    # 5) gm = g + markers (markers only on background cells, no conflict).
    nodes.append(helper.make_node("Add", ["g", "markf"], ["gmf"]))  # f32 [1,1,30,30]
    nodes.append(helper.make_node("Cast", ["gmf"], ["gmc"], to=U8))  # uint8 [1,1,30,30]
    # set padding (out-of-grid) cells to sentinel 255 so they match no color 0..9
    inits.append(cu("sent", 255))
    nodes.append(helper.make_node("Where", ["valid", "gmc", "sent"], ["gm"]))  # uint8 [1,1,30,30]

    # 6) one-hot via Equal against colors 0..9 -> graph output bool [1,10,30,30] (excluded)
    inits.append(cu("colors", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)))
    nodes.append(helper.make_node("Equal", ["gm", "colors"], ["output"]))  # bool [1,10,30,30]

    g = helper.make_graph(
        nodes, "task230",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b17_task230.onnx")
