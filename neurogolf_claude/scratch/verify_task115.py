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
total=0; ok=0; fails=[]
for split in ['train','test','arc-gen']:
    for idx,ex in enumerate(d[split]):
        total+=1
        inp=ex['input']; exp=np.array(ex['output'])
        try:
            pred=solve(inp)
        except Exception as e:
            fails.append((split,idx,'EXC:'+str(e),None,None))
            continue
        if pred.shape==exp.shape and np.array_equal(pred,exp):
            ok+=1
        else:
            fails.append((split,idx,'mismatch',pred.shape,exp.shape))

print(f"TOTAL={total} OK={ok} FRAC={ok/total:.6f}")
print(f"NUM_FAILS={len(fails)}")
for f in fails[:25]:
    print(f)
