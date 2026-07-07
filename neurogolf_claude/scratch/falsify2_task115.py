import json
import numpy as np

def solve(i):
    a=np.array(i); H,W=a.shape
    def runs(seq):
        out=[]
        for v in seq:
            if not out or out[-1]!=v: out.append(int(v))
        return out
    cs=[int(np.bincount(a[:,c]).argmax()) for c in range(W)]
    rs=[int(np.bincount(a[r,:]).argmax()) for r in range(H)]
    cp=sum(np.sum(a[:,c]==np.bincount(a[:,c]).argmax()) for c in range(W))/(H*W)
    rp=sum(np.sum(a[r,:]==np.bincount(a[r,:]).argmax()) for r in range(H))/(H*W)
    if cp>rp: return np.array(runs(cs)).reshape(1,-1)
    return np.array(runs(rs)).reshape(-1,1)

d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task115.json'))

# Stress: how dominant is the per-line majority? If a line is near-tied (lots of noise),
# argmax could pick the wrong color. Find the minimum per-line purity of the CHOSEN axis lines.
min_line_purity=1.0
worst=None
for split in ['train','test','arc-gen']:
    for idx,ex in enumerate(d[split]):
        a=np.array(ex['input']); H,W=a.shape
        cp=sum(np.sum(a[:,c]==np.bincount(a[:,c]).argmax()) for c in range(W))/(H*W)
        rp=sum(np.sum(a[r,:]==np.bincount(a[r,:]).argmax()) for r in range(H))/(H*W)
        if cp>rp:
            # banded along columns -> each column is one band
            for c in range(W):
                col=a[:,c]; bc=np.bincount(col); maj=bc.argmax()
                pur=bc[maj]/H
                if pur<min_line_purity: min_line_purity=pur; worst=(split,idx,'col',c,round(pur,3))
        else:
            for r in range(H):
                row=a[r,:]; bc=np.bincount(row); maj=bc.argmax()
                pur=bc[maj]/W
                if pur<min_line_purity: min_line_purity=pur; worst=(split,idx,'row',r,round(pur,3))
print("min per-line purity on chosen axis:", round(min_line_purity,3), worst)

# Also: does run-collapse ever merge two distinct adjacent bands of same color (would lose info)?
# Compare collapsed-majority length to true number of bands. Check expected output never has
# adjacent equal values (a true RLE collapse property).
viol=0
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        exp=np.array(ex['output']).flatten()
        for k in range(1,len(exp)):
            if exp[k]==exp[k-1]: viol+=1
print("adjacent-equal pairs in expected outputs (should be 0 for clean RLE):", viol)
