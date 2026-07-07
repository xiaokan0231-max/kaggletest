import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# ---- task156 builder (v2: minimized interior memory) ----
# Rule: two solid color-4 rectangles (always row-separated). Peel 1-cell border;
# fill interior of LARGER (area) with 2, SMALLER with 1.
# I/O: input [1,10,30,30] f32 one-hot, grid 10x10 at top-left; out-of-grid one-hot = ALL ZERO.
# Minimization:
#  - One Conv -> color grid [1,1,30,30] f32 (irreducible 3600B input cost).
#  - Slice to 10x10; do all interior logic in uint8/bool (100B tensors).
#  - Equal on 10x10 -> [1,10,10,10] bool one-hot (1000B), then Pad to [1,10,30,30] OUTPUT
#    (output excluded). Avoids the 3600B padded color grid entirely.

g = []
inits = []
def C(name, arr):
    inits.append(numpy_helper.from_array(np.asarray(arr), name)); return name

# We only need the color-4 mask (output 'keep' colors are 0 and 4 = 4*mask).
# Slice input channel 4 AND crop spatially to 10x10 in ONE op -> [1,1,10,10] f32 (400B).
# This avoids materializing any [1,1,30,30] interior tensor.
C('s0', np.array([4,0,0], dtype=np.int64))
C('s_end', np.array([5,10,10], dtype=np.int64))
C('s_ax', np.array([1,2,3], dtype=np.int64))
g.append(helper.make_node('Slice', ['input','s0','s_end','s_ax'], ['ch4']))           # [1,1,10,10] f32
# mask_b = ch4 > 0.5 (one-hot channel 4 is 1.0 where color==4)
C('half_f', np.array(0.5, dtype=np.float32))
g.append(helper.make_node('Greater', ['ch4','half_f'], ['mask_b']))                   # bool

# interior = erosion3x3(mask) via 3x3 box-sum Conv with auto 0-pad.
# boxsum = count of color-4 in 3x3 nbhd (OOB padded 0 => boundary cells not interior).
# interior <=> boxsum == 9.
g.append(helper.make_node('Cast', ['mask_b'], ['maskf16'], to=TensorProto.FLOAT16))   # [1,1,10,10] fp16
C('ones3', np.ones((1,1,3,3), dtype=np.float16))
g.append(helper.make_node('Conv', ['maskf16','ones3'], ['boxsum'], kernel_shape=[3,3], pads=[1,1,1,1]))  # fp16
C('thr_h', np.array(8.5, dtype=np.float16))
g.append(helper.make_node('Greater', ['boxsum','thr_h'], ['interior_b']))             # bool (boxsum==9)

# rowsum over W (axis3): sum maskf16 (fp16 0/1) directly -> fp16 rowsum (areas<=900 exact in fp16)
C('ax3', np.array([3], dtype=np.int64))
g.append(helper.make_node('ReduceSum', ['maskf16','ax3'], ['rowsum'], keepdims=1))    # [1,1,10,1] fp16
C('zero_h', np.array(0.0, dtype=np.float16))
g.append(helper.make_node('Greater', ['rowsum','zero_h'], ['rowhas_b']))              # bool
# CumSum requires float32: build z in f32 from the bool
g.append(helper.make_node('Not', ['rowhas_b'], ['z_b']))                              # bool (gap rows)
g.append(helper.make_node('Cast', ['z_b'], ['z'], to=TensorProto.FLOAT))              # f32 0/1
C('ax2s', np.array(2, dtype=np.int64))
g.append(helper.make_node('CumSum', ['z','ax2s'], ['cumz']))                          # [1,1,10,1] f32
# masked = cumz on content rows, big on gap rows -> min over rows = L (top-block cumz)
C('big', np.array(1000.0, dtype=np.float32))
g.append(helper.make_node('Where', ['rowhas_b','cumz','big'], ['masked']))            # [1,1,10,1] f32
g.append(helper.make_node('ReduceMin', ['masked'], ['Lval'], axes=[2], keepdims=1))   # [1,1,1,1] f32
# is_top_row_b = rowhas & (cumz==L)
g.append(helper.make_node('Equal', ['cumz','Lval'], ['eqL_b']))                       # bool [1,1,10,1]
g.append(helper.make_node('And', ['eqL_b','rowhas_b'], ['is_top_b']))                 # bool [1,1,10,1]
g.append(helper.make_node('Cast', ['is_top_b'], ['is_top_h'], to=TensorProto.FLOAT16))
# top area vs total (fp16)
g.append(helper.make_node('Mul', ['rowsum','is_top_h'], ['rs_top']))
g.append(helper.make_node('ReduceSum', ['rs_top'], ['topA'], keepdims=0))             # scalar fp16
g.append(helper.make_node('ReduceSum', ['rowsum'], ['totalA'], keepdims=0))           # scalar fp16
g.append(helper.make_node('Sub', ['totalA','topA'], ['botA']))
g.append(helper.make_node('Greater', ['topA','botA'], ['topbig_b']))                  # scalar bool
# fill: a row gets color 1 iff (is_top XOR topbig); else color 2.
g.append(helper.make_node('Xor', ['is_top_b','topbig_b'], ['gets1_b']))               # bool [1,1,10,1]
C('one_u', np.array(1, dtype=np.uint8))
C('two_u', np.array(2, dtype=np.uint8))
g.append(helper.make_node('Where', ['gets1_b','one_u','two_u'], ['fillrow_u']))       # uint8 [1,1,10,1]

# keep grid = 4 where mask else 0 (uint8), via Where with scalar consts
C('four_u', np.array(4, dtype=np.uint8))
C('zero_u', np.array(0, dtype=np.uint8))
g.append(helper.make_node('Where', ['mask_b','four_u','zero_u'], ['keep']))           # [1,1,10,10] uint8
# gm = Where(interior_b, fillrow_u, keep)  (uint8, broadcasts fillrow over W)
g.append(helper.make_node('Where', ['interior_b','fillrow_u','keep'], ['gm10']))      # [1,1,10,10] uint8

# Pad gm10 to [1,1,30,30] uint8 with sentinel 255 (out-of-grid: equals no color -> all-zero one-hot)
C('sent_u', np.array(255, dtype=np.uint8))
C('pads30', np.array([0,0,0,0,0,0,20,20], dtype=np.int64))
g.append(helper.make_node('Pad', ['gm10','pads30','sent_u'], ['gm30'], mode='constant'))  # [1,1,30,30] uint8

# final Equal vs colors[1,10,1,1] uint8 -> output one-hot [1,10,30,30] (EXCLUDED)
C('colors_u', np.arange(10, dtype=np.uint8).reshape(1,10,1,1))
g.append(helper.make_node('Equal', ['gm30','colors_u'], ['output']))                  # [1,10,30,30] bool OUTPUT

inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,10,30,30])
out = helper.make_tensor_value_info('output', TensorProto.BOOL, [1,10,30,30])
graph = helper.make_graph(g, 'task156', [inp], [out], inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, '/tmp/p1b_task156.onnx')
print('saved')
