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

# Examine output shapes / orientations
shapes={'row':0,'col':0,'other':0}
cp_rp_margins=[]
close_calls=[]
for split in ['train','test','arc-gen']:
    for idx,ex in enumerate(d[split]):
        inp=np.array(ex['input']); exp=np.array(ex['output'])
        H,W=inp.shape
        if exp.shape[0]==1: shapes['row']+=1
        elif exp.shape[1]==1: shapes['col']+=1
        else: shapes['other']+=1
        cp=sum(np.sum(inp[:,c]==np.bincount(inp[:,c]).argmax()) for c in range(W))/(H*W)
        rp=sum(np.sum(inp[r,:]==np.bincount(inp[r,:]).argmax()) for r in range(H))/(H*W)
        cp_rp_margins.append(abs(cp-rp))
        if abs(cp-rp)<0.05:
            close_calls.append((split,idx,round(cp,3),round(rp,3),exp.shape))

print("output shapes:", shapes)
print("min |cp-rp| margin:", round(min(cp_rp_margins),4))
print("num close calls (<0.05):", len(close_calls))
for c in close_calls[:20]: print("  close:",c)
