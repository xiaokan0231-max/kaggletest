import json, numpy as np

def solve(inp):
    import numpy as np
    inp=np.array(inp)
    ys,xs=np.where(inp==4); r0,r1,c0,c1=ys.min(),ys.max(),xs.min(),xs.max()
    H=r1-r0+1; W=c1-c0+1
    interior=inp[r0:r1+1,c0:c1+1].copy(); interior[interior==4]=0
    m=inp.copy(); m[r0:r1+1,c0:c1+1]=0
    ky,kx=np.where(m>0); key=m[ky.min():ky.max()+1,kx.min():kx.max()+1]
    best=None
    for f in range(1,6):
        up=np.kron(key, np.ones((f,f),dtype=int)); uh,uw=up.shape
        if uh>H or uw>W: continue
        for orr in range(0,H-uh+1):
            for occ in range(0,W-uw+1):
                canvas=np.zeros((H,W),dtype=int); canvas[orr:orr+uh,occ:occ+uw]=up
                mask=interior>0
                if mask.sum()>0 and np.all(canvas[mask]==interior[mask]):
                    sc=int(mask.sum())
                    if best is None or sc>best[0]: best=(sc,canvas)
    out=(best[1] if best else interior).copy()
    out[0,0]=out[0,W-1]=out[H-1,0]=out[H-1,W-1]=4
    return out

d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task209.json'))
exs=list(d.get('train',[]))+list(d.get('test',[]))+list(d.get('arc-gen',[]))
n=len(exs); ok=0; shape_ok=0; fails=[]
for k,ex in enumerate(exs):
    inp=ex['input']; exp=np.array(ex['output'])
    try:
        pred=np.array(solve(inp))
    except Exception as e:
        fails.append((k,'EXC',str(e))); continue
    if pred.shape==exp.shape:
        shape_ok+=1
        if np.array_equal(pred,exp): ok+=1
        else: fails.append((k,'VALDIFF',f'{(pred!=exp).sum()} cells differ'))
    else:
        fails.append((k,'SHAPE',f'pred {pred.shape} exp {exp.shape}'))
print(f'TOTAL={n}  exact_match={ok}  ({ok/n:.4f})  shape_ok={shape_ok} ({shape_ok/n:.4f})')
print('first 15 fails:')
for f in fails[:15]: print(' ', f)
