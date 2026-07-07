import json, numpy as np

def solve(inp):
    inp=np.array(inp)
    m8=(inp==8); yy,xx=np.where(m8); shape=m8[yy.min():yy.max()+1,xx.min():xx.max()+1].astype(int)
    fr=inp.copy(); fr[inp==8]=0; yy2,xx2=np.where(fr>0); r0,r1,c0,c1=yy2.min(),yy2.max(),xx2.min(),xx2.max()
    box=inp[r0:r1+1,c0:c1+1].copy()
    top=int(box[0][box[0]>0][0]); bot=int(box[-1][box[-1]>0][0])
    left=int(box[:,0][box[:,0]>0][0]); right=int(box[:,-1][box[:,-1]>0][0])
    H,W=shape.shape; interior=np.where(shape,8,0)
    for r in range(H):
        for c in range(W):
            if not shape[r,c]: continue
            ds=[(top,r),(bot,H-1-r),(left,c),(right,W-1-c)]
            mn=min(dd for _,dd in ds); win=[col for col,dd in ds if dd==mn]
            if len(win)==1: interior[r,c]=win[0]
    out=box.copy(); out[1:-1,1:-1]=interior
    return out

d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task238.json'))
allex=[]
for split in ['train','test','arc-gen']:
    for ex in d.get(split,[]):
        allex.append((split,ex))

total=len(allex); ok=0; fails=[]; errs=0
for idx,(split,ex) in enumerate(allex):
    inp=ex['input']; exp=np.array(ex['output'])
    try:
        pred=solve(inp)
    except Exception as e:
        errs+=1
        fails.append((idx,split,'EXC',str(e)))
        continue
    if pred.shape==exp.shape and np.array_equal(pred,exp):
        ok+=1
    else:
        reason='shape' if pred.shape!=exp.shape else 'values'
        if len(fails)<8:
            fails.append((idx,split,reason,f'pred{pred.shape} exp{exp.shape}'))

print(f"TOTAL={total} OK={ok} ERRS={errs} FRAC={ok/total:.6f}")
print("FAIL_COUNT=", total-ok)
for f in fails:
    print("FAIL", f)
