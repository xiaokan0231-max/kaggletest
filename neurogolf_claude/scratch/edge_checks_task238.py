import json, numpy as np
d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task238.json'))
allex=[]
for split in ['train','test','arc-gen']:
    for ex in d.get(split,[]):
        allex.append((split,ex))

# Check: 8-shape bbox dims == box interior dims (else replacement would broadcast-error or mismatch)
mismatch_dims=0
frame_row_multicolor=0
frame_uniform_edges=0
for split,ex in allex:
    inp=np.array(ex['input'])
    m8=(inp==8); yy,xx=np.where(m8)
    sh=m8[yy.min():yy.max()+1, xx.min():xx.max()+1]
    H,W=sh.shape
    fr=inp.copy(); fr[inp==8]=0
    yy2,xx2=np.where(fr>0); r0,r1,c0,c1=yy2.min(),yy2.max(),xx2.min(),xx2.max()
    box=inp[r0:r1+1,c0:c1+1]
    bH,bW=box.shape
    if (bH-2,bW-2)!=(H,W): mismatch_dims+=1
    # are the frame edges uniform single color (ignoring the corners/zeros)?
    top_vals=set(box[0][box[0]>0].tolist())
    bot_vals=set(box[-1][box[-1]>0].tolist())
    left_vals=set(box[:,0][box[:,0]>0].tolist())
    right_vals=set(box[:,-1][box[:,-1]>0].tolist())
    if len(top_vals)>1 or len(bot_vals)>1 or len(left_vals)>1 or len(right_vals)>1:
        frame_row_multicolor+=1

print("dim mismatch (8bbox vs interior):", mismatch_dims)
print("frame edge multicolor (edge not single color):", frame_row_multicolor)
print("total:", len(allex))
