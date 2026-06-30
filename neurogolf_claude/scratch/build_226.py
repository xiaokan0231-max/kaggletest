import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# ---- Plan ----
# Input: [1,10,30,30] f32 one-hot. Grid is 10x10 in top-left.
# 1) Collapse 10-channel one-hot -> single color grid g [1,1,30,30] via Conv weight [0..9].
# 2) Crop to 10x10 -> work tiny on uint8/bool.
# 3) Rule: full-gray rows/cols split into bands. 3 colors at proportional band indices.
#    band-row-index = cumulative count of gray rows strictly above (exclusive prefix sum).
#    R = total gray rows + 1 (3 or 5). targets per color k: tr=k*(R-1)//2, same for cols.
#    color1: (rb=0, cb=0); color2: (rb=(R-1)/2, cb=(C-1)/2); color3: (rb=R-1, cb=C-1)
#    fill background-0 cells in the targeted band-cell with color k.
# 4) Build output color grid gm [1,1,10,10] uint8, pad to 30x30 with sentinel 255 (so Equal->all false there => all-zero one-hot, matching out-of-grid).
# 5) Equal(gm30 broadcast, colors[1,10,1,1] uint8) -> output bool [1,10,30,30] (graph output, excluded).

nodes=[]; inits=[]
def C(name, arr):
    t=numpy_helper.from_array(np.asarray(arr), name=name)
    inits.append(t); return name

# 1) collapse to color grid
Wcol = np.arange(10, dtype=np.float32).reshape(1,10,1,1)
C('Wcol', Wcol)
nodes.append(helper.make_node('Conv',['input','Wcol'],['g30f'],kernel_shape=[1,1]))  # [1,1,30,30] f32

# 2) crop to 10x10
C('s0', np.array([0,0],dtype=np.int64))
C('e0', np.array([10,10],dtype=np.int64))
C('ax', np.array([2,3],dtype=np.int64))
nodes.append(helper.make_node('Slice',['g30f','s0','e0','ax'],['g10f']))  # [1,1,10,10] f32

# cast to uint8 color grid  (colors 0..9, gray=5)
nodes.append(helper.make_node('Cast',['g10f'],['g'],to=TensorProto.UINT8))  # [1,1,10,10] u8

# ---- gray masks ----
# graycell = (g==5)
C('five', np.array(5,dtype=np.uint8))
nodes.append(helper.make_node('Equal',['g','five'],['grayb']))  # bool [1,1,10,10]
nodes.append(helper.make_node('Cast',['grayb'],['grayf'],to=TensorProto.FLOAT))

# full gray row: a row is all-gray if sum over width of grayf == 10
# Conv over width with weight ones[1,1,1,10] -> per-row count [1,1,10,1]
C('Wrow', np.ones((1,1,1,10),dtype=np.float32))
nodes.append(helper.make_node('Conv',['grayf','Wrow'],['rowcnt']))   # [1,1,10,1]
C('Wcolk', np.ones((1,1,10,1),dtype=np.float32))
nodes.append(helper.make_node('Conv',['grayf','Wcolk'],['colcnt']))   # [1,1,1,10]
# grayrow indicator (per row) = rowcnt==10
C('ten', np.array(10,dtype=np.float32))
nodes.append(helper.make_node('Equal',['rowcnt','ten'],['growb']))   # [1,1,10,1] bool
nodes.append(helper.make_node('Equal',['colcnt','ten'],['gcolb']))   # [1,1,1,10] bool
nodes.append(helper.make_node('Cast',['growb'],['growf'],to=TensorProto.FLOAT))  # [1,1,10,1]
nodes.append(helper.make_node('Cast',['gcolb'],['gcolf'],to=TensorProto.FLOAT))  # [1,1,1,10]

# R = sum(growf)+1 ; C = sum(gcolf)+1   (scalars)
nodes.append(helper.make_node('ReduceSum',['growf'],['Rg'],keepdims=0))  # scalar = #gray rows
nodes.append(helper.make_node('ReduceSum',['gcolf'],['Cg'],keepdims=0))
# (R-1)=Rg ; (R-1)/2 = Rg/2 ; R-1 = Rg
C('two', np.array(2.0,dtype=np.float32))
nodes.append(helper.make_node('Div',['Rg','two'],['Rhalf']))  # (R-1)/2  scalar float
nodes.append(helper.make_node('Div',['Cg','two'],['Chalf']))

# band-row-index for each row = exclusive prefix sum of growf along H.
# exclusive prefix = cumsum(inclusive) - growf.  Use CumSum (opset 11+).
C('axisH', np.array(2,dtype=np.int64))
C('axisW', np.array(3,dtype=np.int64))
nodes.append(helper.make_node('CumSum',['growf','axisH'],['rincl']))  # inclusive [1,1,10,1]
nodes.append(helper.make_node('Sub',['rincl','growf'],['ridx']))      # exclusive band index per row [1,1,10,1]
nodes.append(helper.make_node('CumSum',['gcolf','axisW'],['cincl']))
nodes.append(helper.make_node('Sub',['cincl','gcolf'],['cidx']))      # [1,1,1,10]

# background-0 mask
C('zero', np.array(0,dtype=np.uint8))
nodes.append(helper.make_node('Equal',['g','zero'],['bg0b']))   # bool [1,1,10,10]

# For each color k, target band indices tr_k (row) and tc_k (col):
#  k=1: tr=0, tc=0
#  k=2: tr=Rhalf, tc=Chalf
#  k=3: tr=Rg,    tc=Cg
# cell in target band: (ridx==tr) broadcast with (cidx==tc) AND bg0
# Build color grid: start with g (keep original), overwrite background cells.
# We'll compute gm = g where not filled, else color.
# Build per-color fill mask then combine.

def band_mask(ridx_eq_val, cidx_eq_val, rowtgt, coltgt, prefix):
    # rmask: ridx==rowtgt  -> bool [1,1,10,1]
    nodes.append(helper.make_node('Equal',['ridx',rowtgt],[prefix+'rm']))
    nodes.append(helper.make_node('Equal',['cidx',coltgt],[prefix+'cm']))
    # combine via And with broadcast -> [1,1,10,10]
    nodes.append(helper.make_node('And',[prefix+'rm',prefix+'cm'],[prefix+'rc']))
    nodes.append(helper.make_node('And',[prefix+'rc','bg0b'],[prefix+'msk']))
    return prefix+'msk'

# targets as float tensors matching ridx dtype (float)
C('f0', np.array(0.0,dtype=np.float32))
m1=band_mask(None,None,'f0','f0','c1_')          # color1 mask
m2=band_mask(None,None,'Rhalf','Chalf','c2_')    # color2 mask
m3=band_mask(None,None,'Rg','Cg','c3_')          # color3 mask

# Build output color grid gm (uint8): start = g, then Where(mask, colorK, prev)
C('col1', np.array(1,dtype=np.uint8))
C('col2', np.array(2,dtype=np.uint8))
C('col3', np.array(3,dtype=np.uint8))
nodes.append(helper.make_node('Where',[m1,'col1','g'],['gm1']))
nodes.append(helper.make_node('Where',[m2,'col2','gm1'],['gm2']))
nodes.append(helper.make_node('Where',[m3,'col3','gm2'],['gm']))   # [1,1,10,10] u8

# Pad gm to [1,1,30,30] with sentinel 255 so out-of-grid -> all-false one-hot
C('pads', np.array([0,0,0,0, 0,0,20,20],dtype=np.int64))  # pad H,W end by 20
C('sent', np.array(255,dtype=np.uint8))
nodes.append(helper.make_node('Pad',['gm','pads','sent'],['gm30'],mode='constant'))  # [1,1,30,30] u8

# Equal with colors [1,10,1,1] uint8 0..9 -> output [1,10,30,30] bool
colors=np.arange(10,dtype=np.uint8).reshape(1,10,1,1)
C('colors',colors)
nodes.append(helper.make_node('Equal',['gm30','colors'],['output']))

graph=helper.make_graph(nodes,'task226',
    [helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,10,30,30])],
    [helper.make_tensor_value_info('output',TensorProto.BOOL,[1,10,30,30])],
    inits)
m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])
m.ir_version=8
onnx.checker.check_model(m)
onnx.save(m,'/tmp/p1e_task226.onnx')
print('saved, nodes=',len(nodes))
