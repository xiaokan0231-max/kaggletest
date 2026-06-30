"""Build compact ONNX for task224.

Rule (verified 266/266): grid holds one solid shape of color M plus four
isolated '5' markers. Draw a 1-cell-thick rectangle OUTLINE of color M whose
bounds are the bounding box of the four 5s inset by 1 on every side. The outline
only ever lands on background cells.

Ultra-compact design (pattern (b)): the output equals the input everywhere
except border cells, which become the one-hot of color M.  We never build a
color grid -- M and five-presence both come straight off the input via tiny
Convs / channel reductions.  The ONLY [1,10,30,30] tensor is the EXCLUDED output.

  rowf = Conv(input, picks ch5, kernel 1x30) -> [1,1,30,1] five-count per row
  colf = Conv(input, picks ch5, kernel 30x1) -> [1,1,1,30] five-count per col
  inrow[r] = (#five-rows above r >0) & (below >0)   via CumSum (f16, 60B each)
  incol likewise
  redge / cedge  = first/last filled row/col via 1-cell shifts
  border = (redge & incol) | (inrow & cedge)        [1,1,30,30] bool  (3 big)
  cnt = ReduceSum(input,[2,3]) [1,10,1,1] ; present = cnt>0
  m_onehot = present & [0,1,1,1,1,0,1,1,1,1]         one-hot of M  [1,10,1,1]
  output = Where(border, m_onehot, input)            f32 [1,10,30,30] == output
"""
import sys
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
N = 30


def cf(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float32), name=name)


def cu(name, v):
    return numpy_helper.from_array(np.asarray(v, np.uint8), name=name)


def ci(name, v):
    return numpy_helper.from_array(np.asarray(v, np.int64), name=name)


def cbz(name, v):
    return numpy_helper.from_array(np.asarray(v, np.bool_), name=name)


def c16(name, v):
    return numpy_helper.from_array(np.asarray(v, np.float16), name=name)


def build(path):
    nodes, inits = [], []

    # 1) five presence per row/col (tiny Conv each, NO big tensor)
    wr = np.zeros((1, 10, 1, N), np.float32); wr[0, 5, 0, :] = 1.0
    wc = np.zeros((1, 10, N, 1), np.float32); wc[0, 5, :, 0] = 1.0
    inits.append(cf("Wrow", wr)); inits.append(cf("Wcol5", wc))
    nodes.append(helper.make_node("Conv", ["input", "Wrow"], ["rowf"]))   # f32 [1,1,30,1]
    nodes.append(helper.make_node("Conv", ["input", "Wcol5"], ["colf"]))  # f32 [1,1,1,30]

    # 2) inrow / incol via cumsum in f16 of the raw five-COUNT per row/col.
    #    We only ever test ">0", so raw counts behave like presence (no clamp).
    nodes.append(helper.make_node("Cast", ["rowf"], ["rowi"], to=F16))   # f16 [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["colf"], ["coli"], to=F16))
    inits.append(ci("ax2", 2)); inits.append(ci("ax3", 3))
    nodes.append(helper.make_node("CumSum", ["rowi", "ax2"], ["rcum"]))
    nodes.append(helper.make_node("CumSum", ["coli", "ax3"], ["ccum"]))
    nodes.append(helper.make_node("Sub", ["rcum", "rowi"], ["rabove"]))
    nodes.append(helper.make_node("Sub", ["ccum", "coli"], ["cabove"]))
    inits.append(ci("rax", [2])); inits.append(ci("cax", [3]))
    nodes.append(helper.make_node("ReduceSum", ["rowi", "rax"], ["rtot"], keepdims=1))
    nodes.append(helper.make_node("ReduceSum", ["coli", "cax"], ["ctot"], keepdims=1))
    nodes.append(helper.make_node("Sub", ["rtot", "rcum"], ["rbelow"]))
    nodes.append(helper.make_node("Sub", ["ctot", "ccum"], ["cbelow"]))
    inits.append(c16("h16", 0.5))
    nodes.append(helper.make_node("Greater", ["rabove", "h16"], ["rap"]))
    nodes.append(helper.make_node("Greater", ["rbelow", "h16"], ["rbp"]))
    nodes.append(helper.make_node("And", ["rap", "rbp"], ["inrow"]))      # bool [1,1,30,1]
    nodes.append(helper.make_node("Greater", ["cabove", "h16"], ["cap"]))
    nodes.append(helper.make_node("Greater", ["cbelow", "h16"], ["cbp"]))
    nodes.append(helper.make_node("And", ["cap", "cbp"], ["incol"]))      # bool [1,1,1,30]

    # 3) redge / cedge via 1-cell shifts (all 30-elem)
    inits.append(cbz("Fpad", [False]))
    inits.append(ci("s1", [1])); inits.append(ci("s30", [30])); inits.append(ci("s0", [0]))
    inits.append(ci("sax2", [2])); inits.append(ci("sax3", [3]))
    # rows
    nodes.append(helper.make_node("Slice", ["inrow", "s1", "s30", "sax2"], ["inrow_u0"]))
    inits.append(ci("padR_end", [0, 0, 0, 0, 0, 0, 1, 0]))
    nodes.append(helper.make_node("Pad", ["inrow_u0", "padR_end", "Fpad"], ["inrow_up"]))
    inits.append(ci("padR_beg", [0, 0, 1, 0, 0, 0, 0, 0]))
    nodes.append(helper.make_node("Pad", ["inrow", "padR_beg", "Fpad"], ["inrow_d0"]))
    nodes.append(helper.make_node("Slice", ["inrow_d0", "s0", "s30", "sax2"], ["inrow_dn"]))
    nodes.append(helper.make_node("Not", ["inrow_up"], ["nup_r"]))
    nodes.append(helper.make_node("Not", ["inrow_dn"], ["ndn_r"]))
    nodes.append(helper.make_node("Or", ["nup_r", "ndn_r"], ["redge0"]))
    nodes.append(helper.make_node("And", ["inrow", "redge0"], ["redge"]))   # bool [1,1,30,1]
    # cols
    nodes.append(helper.make_node("Slice", ["incol", "s1", "s30", "sax3"], ["incol_u0"]))
    inits.append(ci("padC_end", [0, 0, 0, 0, 0, 0, 0, 1]))
    nodes.append(helper.make_node("Pad", ["incol_u0", "padC_end", "Fpad"], ["incol_up"]))
    inits.append(ci("padC_beg", [0, 0, 0, 1, 0, 0, 0, 0]))
    nodes.append(helper.make_node("Pad", ["incol", "padC_beg", "Fpad"], ["incol_d0"]))
    nodes.append(helper.make_node("Slice", ["incol_d0", "s0", "s30", "sax3"], ["incol_dn"]))
    nodes.append(helper.make_node("Not", ["incol_up"], ["nup_c"]))
    nodes.append(helper.make_node("Not", ["incol_dn"], ["ndn_c"]))
    nodes.append(helper.make_node("Or", ["nup_c", "ndn_c"], ["cedge0"]))
    nodes.append(helper.make_node("And", ["incol", "cedge0"], ["cedge"]))   # bool [1,1,1,30]

    # 4) border via level product:  rlev,clev in {0,1,2}  (0=outside, 1=interior
    #    filled, 2=edge).  border <=> rlev*clev >= 2  -> only TWO big tensors.
    nodes.append(helper.make_node("Cast", ["inrow"], ["inrow_u"], to=U8))   # [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["redge"], ["redge_u"], to=U8))
    nodes.append(helper.make_node("Add", ["inrow_u", "redge_u"], ["rlev"]))  # u8 [1,1,30,1]
    nodes.append(helper.make_node("Cast", ["incol"], ["incol_u"], to=U8))   # [1,1,1,30]
    nodes.append(helper.make_node("Cast", ["cedge"], ["cedge_u"], to=U8))
    nodes.append(helper.make_node("Add", ["incol_u", "cedge_u"], ["clev"]))  # u8 [1,1,1,30]
    nodes.append(helper.make_node("Mul", ["rlev", "clev"], ["prod"]))        # u8 [1,1,30,30]
    inits.append(cu("u1", 1))
    nodes.append(helper.make_node("Greater", ["prod", "u1"], ["border"]))    # bool [1,1,30,30]

    # 5) m_onehot = present & not-{0,5}   (one-hot of color M, all tiny)
    inits.append(ci("rax23", [2, 3]))
    inits.append(cf("h0", 0.5))
    nodes.append(helper.make_node("ReduceSum", ["input", "rax23"], ["cnt"], keepdims=1))  # f32 [1,10,1,1]
    nodes.append(helper.make_node("Greater", ["cnt", "h0"], ["present"]))                 # bool [1,10,1,1]
    notmask = np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1], np.bool_).reshape(1, 10, 1, 1)
    inits.append(cbz("notmask", notmask))
    nodes.append(helper.make_node("And", ["present", "notmask"], ["m_oh_b"]))             # bool [1,10,1,1]
    nodes.append(helper.make_node("Cast", ["m_oh_b"], ["m_onehot"], to=F32))              # f32 [1,10,1,1]

    # 6) output = Where(border, m_onehot, input)  -- only [1,10,30,30] is the output
    nodes.append(helper.make_node("Where", ["border", "m_onehot", "input"], ["output"]))  # f32 [1,10,30,30]

    x_in = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    graph = helper.make_graph(nodes, "t224", [x_in], [y], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print("saved", path, "nodes", len(nodes))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b20_task224.onnx")
