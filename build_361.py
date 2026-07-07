"""Compact ONNX for task361: C4-symmetrize input about score-argmax center."""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

N = 10
D = 2 * N        # 20
P = 44           # padded canvas for translation pipeline
A = 22           # anchor (doubled center) in padded canvas
SM = 3           # separation half-window for score stacks
ks = 2 * SM + 1
K = ks * ks
offs = [(orr, occ) for orr in range(-SM, SM + 1) for occ in range(-SM, SM + 1)]

nodes, inits = [], []


def C(name, arr, dtype=None):
    if dtype is not None:
        arr = arr.astype(dtype)
    inits.append(numpy_helper.from_array(arr, name))
    return name


def nd(op, ins, outs, **attrs):
    nodes.append(helper.make_node(op, ins, outs, **attrs))
    return outs[0]


# ---------- 1. collapse one-hot -> color grid, crop to 10x10 ----------
C("w_collapse", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nd("Conv", ["input", "w_collapse"], ["g30"])
C("s0", np.array([0, 0, 0, 0], np.int64))
C("e10", np.array([1, 1, N, N], np.int64))
C("ax4", np.array([0, 1, 2, 3], np.int64))
nd("Slice", ["g30", "s0", "e10", "ax4"], ["g"])           # [1,1,10,10] f32

C("sh1010", np.array([N, N], np.int64))
nd("Reshape", ["g", "sh1010"], ["g2d"])                   # [10,10]
C("zero", np.array(0.0, np.float32))
nd("Greater", ["g2d", "zero"], ["nzb"])
nd("Cast", ["nzb"], ["nz2d"], to=TensorProto.FLOAT)
C("sh1110", np.array([1, 1, N, N], np.int64))
nd("Reshape", ["nz2d", "sh1110"], ["nz"])                 # [1,1,10,10]

# ---------- shift kernels (shared) ----------
# Conv on padded g with one-hot kernels -> [1,K,10,10] shifts.
Wshift = np.zeros((K, 1, ks, ks), np.float32)
for j, (orr, occ) in enumerate(offs):
    Wshift[j, 0, orr + SM, occ + SM] = 1.0   # out[r,c]=gp[r+(orr+SM), c+(occ+SM)] = g[r+orr, c+occ]
C("Wshift", Wshift)
C("pads_g", np.array([0, 0, SM, SM, 0, 0, SM, SM], np.int64))

# =========================================================
# S180 in (k,l):  center k=2r+orr, l=2c+occ  (separable stride-2)
# =========================================================
nd("Pad", ["g", "pads_g"], ["gp"])
nd("Conv", ["gp", "Wshift"], ["shift180"])               # [1,K,10,10]
nd("Equal", ["shift180", "g"], ["eq180"])                # broadcast g
nd("Cast", ["eq180"], ["eq180f"], to=TensorProto.FLOAT)
nd("Mul", ["eq180f", "nz"], ["m180"])                    # [1,K,10,10] match
# placement: out[2r+orr, 2c+occ] += m180[k_ch,r,c]; ConvTranspose stride2, kernel offset (orr+SM, occ+SM)
Wp180 = np.zeros((K, 1, ks, ks), np.float32)
for j, (orr, occ) in enumerate(offs):
    Wp180[j, 0, orr + SM, occ + SM] = 1.0
C("Wp180", Wp180)
nd("ConvTranspose", ["m180", "Wp180"], ["S180big"], strides=[2, 2])
# S180big spatial = 2*(10-1)+ks = 18+ks ; index = (2r+orr)+SM ; read [SM:SM+D]
C("ss", np.array([0, 0, SM, SM], np.int64))
C("se", np.array([1, 1, SM + D, SM + D], np.int64))
nd("Slice", ["S180big", "ss", "se", "ax4"], ["S180"])    # [1,1,20,20] in (k,l)

# =========================================================
# S90 in (u,w): u = 2c + occ, w = (N-1) - 2r - orr   (separable)
#   partner = g90 shifted; g90[a,b] = g[b, N-1-a]
# =========================================================
nd("Transpose", ["g2d"], ["gT"], perm=[1, 0])            # gT[c,r]=g[r,c]
C("flipidx", np.arange(N - 1, -1, -1, dtype=np.int64))
nd("Gather", ["gT", "flipidx"], ["g90"], axis=0)         # g90[a,b]=g[b,N-1-a]
nd("Reshape", ["g90", "sh1110"], ["g90_4"])
nd("Pad", ["g90_4", "pads_g"], ["g90p"])
nd("Conv", ["g90p", "Wshift"], ["shift90"])              # [1,K,10,10] = g90 shifted by offsets
nd("Equal", ["shift90", "g"], ["eq90"])
nd("Cast", ["eq90"], ["eq90f"], to=TensorProto.FLOAT)
nd("Mul", ["eq90f", "nz"], ["m90"])                      # match in (r,c) for partner g90[r+orr,c+occ]
# placement into (u,w):  u = 2c + occ ; w = (N-1) - 2r - orr
#   stride-2 over c gives 2c (axis w-of-image is columns=u), over r gives 2r.
#   For w we need (N-1)-2r-orr: flip r-axis first so r'=N-1-r => 2r' = 2(N-1)-2r ; w = 2r' -(N-1) -orr.
#   We'll place with ConvTranspose stride2 producing rows=2r', cols=2c, then slice/offset to align.
# Flip m90 along the H (r) axis:
C("flipidxK", np.arange(N - 1, -1, -1, dtype=np.int64))
nd("Gather", ["m90", "flipidxK"], ["m90f"], axis=2)      # flip rows: m90f[k,r',c]=m90[k,N-1-r',c]
# Now place: outrow = 2*r' + (orr offset), outcol = 2*c + (occ offset).
#   we want final w = 2r' - (N-1) - orr ; final u = 2c + occ.
#   ConvTranspose: out[2r'+i, 2c+j] += m90f[k,r',c] * Wp90[k,0,i,j]; choose i = (SM - orr)?? to encode -orr.
# We want the orr contribution to be -orr in w. With outrow0 = 2r'+i, and w = outrow0 + const,
#   set i = (SM - orr) so that placement row carries -orr (plus const SM). Then w = (2r' + SM - orr) + base.
Wp90 = np.zeros((K, 1, ks, ks), np.float32)
for j, (orr, occ) in enumerate(offs):
    Wp90[j, 0, SM - orr, occ + SM] = 1.0
C("Wp90", Wp90)
nd("ConvTranspose", ["m90f", "Wp90"], ["S90big"], strides=[2, 2])
# S90big rows index = 2r' + (SM - orr) ; we want w' coordinate. We'll crop to DxD and let calibration handle
# the constant origin via the (u,w)->(k,l) gather built from numpy.
C("se2", np.array([1, 1, ks - 1 + 2 * (N - 1) + 1, ks - 1 + 2 * (N - 1) + 1], np.int64))
nd("Slice", ["S90big", "ss", "se2", "ax4"], ["S90uw_full"])
# We'll select a DxD window [0:D,0:D] of the (rows=w-ish, cols=u) map.
C("se3", np.array([1, 1, D, D], np.int64))
nd("Slice", ["S90uw_full", "s0", "se3", "ax4"], ["S90uw"])   # [1,1,20,20] in (wrow, ucol)

# remap (wrow,ucol) -> (k,l) via constant Gather index map (built in numpy, calibrated).
# placeholder: GatherND with a [20,20,2] index; computed in calibration step below.
C("remap_idx", np.zeros((D, D, 2), np.int64))   # will be overwritten before save
nd("Reshape", ["S90uw", np.array(0)], ["_dummy"]) if False else None

# We'll instead do remap with GatherND on flattened. Build flat S90uw [400] and gather.
C("flat400", np.array([1, D * D], np.int64))
nd("Reshape", ["S90uw", "flat400"], ["S90flat"])         # [1,400]
# gather indices [400] (for each (k,l) flat -> source uw flat); constant, calibrated.
C("g90map", np.zeros((D * D,), np.int64))
nd("Gather", ["S90flat", "g90map"], ["S90kl_g"], axis=1)  # [1,400]
C("shkl", np.array([1, 1, D, D], np.int64))
nd("Reshape", ["S90kl_g", "shkl"], ["S90"])              # [1,1,20,20] in (k,l)

# score = 2*S90 + S180
C("two", np.array(2.0, np.float32))
nd("Mul", ["S90", "two"], ["S90x2"])
nd("Add", ["S90x2", "S180"], ["score"])                  # [1,1,20,20]

# =========================================================
# argmax over flattened score -> center index -> (cr2,cc2)
# =========================================================
nd("Reshape", ["score", "flat400"], ["scoreflat"])       # [1,400]
nd("ArgMax", ["scoreflat"], ["amax"], axis=1, keepdims=0)  # [1] int64 flat index
# cr2 = idx // D ; cc2 = idx % D
C("Dconst", np.array([D], np.int64))
nd("Div", ["amax", "Dconst"], ["cr2"])
nd("Mod", ["amax", "Dconst"], ["cc2"])

# =========================================================
# OUTPUT: translate doubled g to anchor, rotate (4x), symmetrize, translate back, downsample
# =========================================================
# doubled grid on padded canvas C0 [P,P]: place g2d at (2r,2c).
# Build by ConvTranspose stride2 on g (identity 1x1 kernel) -> [1,1,19,19], then pad to P.
C("idker", np.ones((1, 1, 1, 1), np.float32))
nd("ConvTranspose", ["g", "idker"], ["gdoub"], strides=[2, 2])   # [1,1,19,19], gdoub[2r,2c]=g
C("padcanvas", np.array([0, 0, 0, 0, 0, 0, P - 19, P - 19], np.int64))
nd("Pad", ["gdoub", "padcanvas"], ["canvas"])            # [1,1,P,P]
nd("Reshape", ["canvas", np.array(0)], ["_d2"]) if False else None
C("shPP", np.array([P, P], np.int64))
nd("Reshape", ["canvas", "shPP"], ["C0"])                # [P,P]

# shift matrices built from data-dependent center.
# rowshift by t = A - cr2 ; colshift by tc = A - cc2.
# shiftmat M[i,j] = 1 if j == i - t  -> (M@v)[i]=v[i-t].
# Build via Equal(rowidx[:,None] - t, colidx[None,:]).
C("idxP", np.arange(P, dtype=np.int64))
# t = A - cr2 (scalar). compute:
C("Aconst", np.array(A, np.int64))
nd("Sub", ["Aconst", "cr2"], ["tr"])     # scalar
nd("Sub", ["Aconst", "cc2"], ["tc"])
# rowidx - tr : need broadcasting. idxP [P]; tr scalar -> idxP - tr [P]
nd("Sub", ["idxP", "tr"], ["rowm_tr"])   # [P] = i - tr
# Mr[i,j] = (rowm_tr[i] == idxP[j])
C("shP1", np.array([P, 1], np.int64))
C("sh1P", np.array([1, P], np.int64))
nd("Reshape", ["rowm_tr", "shP1"], ["rowm_tr_col"])      # [P,1]
nd("Reshape", ["idxP", "sh1P"], ["idx_row"])             # [1,P]
nd("Equal", ["rowm_tr_col", "idx_row"], ["Mr_b"])        # [P,P]
nd("Cast", ["Mr_b"], ["Mr"], to=TensorProto.FLOAT)
nd("Sub", ["idxP", "tc"], ["rowm_tc"])
nd("Reshape", ["rowm_tc", "shP1"], ["rowm_tc_col"])
nd("Equal", ["rowm_tc_col", "idx_row"], ["Mc_b"])
nd("Cast", ["Mc_b"], ["Mc"], to=TensorProto.FLOAT)

# translate: C1 = Mr @ C0 @ Mc^T   (rows shifted by tr, cols by tc)
nd("MatMul", ["Mr", "C0"], ["tmp1"])
nd("Transpose", ["Mc"], ["McT"], perm=[1, 0])
nd("MatMul", ["tmp1", "McT"], ["C1"])                    # [P,P], center now at (A,A)

# rotations about anchor A (static). Build reversal matrix Rrev[i,j]=1 if j==2A-i.
C("twoA", np.array(2 * A, np.int64))
nd("Sub", ["twoA", "idxP"], ["rev_i"])     # 2A - i  [P]
nd("Reshape", ["rev_i", "shP1"], ["rev_i_col"])
nd("Equal", ["rev_i_col", "idx_row"], ["Rrev_b"])
nd("Cast", ["Rrev_b"], ["Rrev"], to=TensorProto.FLOAT)   # [P,P], constant-ish (data-indep!) but fine

# r180 = Rrev @ C1 @ Rrev^T
nd("MatMul", ["Rrev", "C1"], ["t180a"])
nd("MatMul", ["t180a", "Rrev"], ["C180"])   # Rrev symmetric so Rrev^T=Rrev
# r90: new[i,j]=C1[2A-j, i]  => first transpose C1 -> Ct[i,j]=C1[j,i]; then?
#   want R90[i,j]=C1[2A-j, i]. = (Rrev @ C1)[? ]...
#   Let X = C1. R90[i,j]=X[2A-j, i]. Define Y=transpose(X): Y[a,b]=X[b,a]. Then X[2A-j,i]=Y[i,2A-j].
#   So R90[i,j]=Y[i, 2A-j] = (Y @ Rrev)[i,j]  since (Y@Rrev)[i,j]=sum_k Y[i,k]Rrev[k,j]=Y[i,2A-j].
nd("Transpose", ["C1"], ["C1T"], perm=[1, 0])
nd("MatMul", ["C1T", "Rrev"], ["C90"])
# r270: new[i,j]=C1[j, 2A-i]. = Y'? X[j,2A-i]. Let Z=transpose? X[j,2A-i]:
#   (Rrev @ X)[i,j]... (Rrev@X)[i,j]=X[2A-i,j]. transpose of that: =X[2A-j? ] messy.
#   X[j,2A-i] = transpose(X)[2A-i, j] = (Rrev @ X^T)[i,j].
nd("MatMul", ["Rrev", "C1T"], ["C270"])

# symmetrize with priority C1 > C90 > C180 > C270 (keep first nonzero).
def overlay(base, layer, outname, maskname):
    nd("Equal", [base, "zeroPP"], [maskname + "_zb"])     # base==0
    nd("Cast", [maskname + "_zb"], [maskname + "_z"], to=TensorProto.FLOAT)
    nd("Mul", [layer, maskname + "_z"], [maskname + "_add"])
    nd("Add", [base, maskname + "_add"], [outname])
C("zeroPP", np.zeros((P, P), np.float32))
overlay("C1", "C90", "O1", "ov1")
overlay("O1", "C180", "O2", "ov2")
overlay("O2", "C270", "O3", "ov3")    # symmetrized in anchor frame

# translate back: shift rows by -tr, cols by -tc => use Mr^T, Mc^T (inverse of shift is transpose)
nd("Transpose", ["Mr"], ["MrT"], perm=[1, 0])
nd("MatMul", ["MrT", "O3"], ["b1"])
nd("MatMul", ["b1", "Mc"], ["B"])     # back-translated [P,P]

# downsample: take rows/cols 0,2,4,...,2*(N-1) -> [N,N]
C("evenidx", np.arange(0, 2 * N, 2, dtype=np.int64))
nd("Gather", ["B", "evenidx"], ["Brow"], axis=0)   # [N,P]
nd("Gather", ["Brow", "evenidx"], ["ans2d"], axis=1)  # [N,N]

# ---------- build 30x30 answer color grid then Equal -> one-hot output ----------
C("shN", np.array([1, 1, N, N], np.int64))
nd("Reshape", ["ans2d", "shN"], ["ans"])           # [1,1,10,10]
C("pad_to30", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64))
nd("Pad", ["ans", "pad_to30"], ["ans30"])          # [1,1,30,30] color grid (0 outside)
C("colors", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
nd("Equal", ["ans30", "colors"], ["out_b"])        # [1,10,30,30] bool = output
nd("Cast", ["out_b"], ["output"], to=TensorProto.FLOAT)

graph = helper.make_graph(
    nodes, "task361",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save(model, "/tmp/b15_task361.onnx")
print("saved scaffold (uncalibrated remap)")
