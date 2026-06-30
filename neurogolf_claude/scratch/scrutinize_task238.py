import json, numpy as np

d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task238.json'))
allex=[]
for split in ['train','test','arc-gen']:
    for ex in d.get(split,[]):
        allex.append((split,ex))

# Diagnostics: check assumptions hold across all examples
ties_seen=0
tie_examples=0
box_color_counts=set()
shape_sizes=set()
output_color_sets=set()
multi_8_components=0
box_has_8_in_frame=0
frame_minus8_connected_issue=0

from scipy import ndimage
for idx,(split,ex) in enumerate(allex):
    inp=np.array(ex['input']); exp=np.array(ex['output'])
    m8=(inp==8)
    yy,xx=np.where(m8); shape=m8[yy.min():yy.max()+1,xx.min():xx.max()+1].astype(int)
    H,W=shape.shape
    fr=inp.copy(); fr[inp==8]=0
    yy2,xx2=np.where(fr>0); r0,r1,c0,c1=yy2.min(),yy2.max(),xx2.min(),xx2.max()
    box=inp[r0:r1+1,c0:c1+1].copy()
    # does box (the frame region) contain any 8? (would corrupt top/bot/left/right detection)
    if (box==8).any():
        box_has_8_in_frame+=1
    top=int(box[0][box[0]>0][0]); bot=int(box[-1][box[-1]>0][0])
    left=int(box[:,0][box[:,0]>0][0]); right=int(box[:,-1][box[:,-1]>0][0])
    box_color_counts.add((top,bot,left,right))
    shape_sizes.add((H-2,W-2))  # interior size
    output_color_sets.add(tuple(sorted(set(exp.flatten().tolist()))))
    # count ties and whether any 8 remains in expected output
    has_tie=False
    for r in range(H):
        for c in range(W):
            if not shape[r,c]: continue
            ds=[r,H-1-r,c,W-1-c]
            mn=min(ds)
            if sum(1 for dd in ds if dd==mn)>1:
                has_tie=True; ties_seen+=1
    if has_tie: tie_examples+=1
    # number of 8 connected components in shape
    lbl,n=ndimage.label(m8)
    if n>1: multi_8_components+=1

print("examples:", len(allex))
print("box_has_8_in_frame:", box_has_8_in_frame)
print("tie_examples (have at least one tie cell):", tie_examples, "total tie cells:", ties_seen)
print("distinct (top,bot,left,right) tuples:", len(box_color_counts), list(box_color_counts)[:10])
print("distinct interior sizes:", sorted(shape_sizes))
print("multi_8_components examples:", multi_8_components)
print("num distinct output color sets:", len(output_color_sets))
for s in sorted(output_color_sets):
    print("   colors:", s)

# Verify: do any expected outputs contain an 8 (i.e. a real tie that stays 8)?
ex_with_8=0
for idx,(split,ex) in enumerate(allex):
    exp=np.array(ex['output'])
    if (exp==8).any(): ex_with_8+=1
print("expected outputs containing an 8:", ex_with_8)
