import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
from collections import deque
d=ng.load_examples(233)
S8=np.ones((3,3),int)
def find_canvas(a):
    mask=(a==2); lbl,n=ndimage.label(mask); best=None;bs=-1
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i); h=ys.max()-ys.min()+1;w=xs.max()-xs.min()+1
        if h*w>bs: bs=h*w;best=(ys.min(),ys.max(),xs.min(),xs.max())
    return best
def get_keys(a,box):
    r0,r1,c0,c1=box; nz=(a!=0); nz[r0:r1+1,c0:c1+1]=False
    lbl,n=ndimage.label(nz,structure=S8); res=[]
    H,W=a.shape
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i); color=int(a[ys[0],xs[0]])
        if color in (0,2): continue
        start=(ys[0],xs[0]); q=deque([start]); reg=set([start])
        while q:
            cy,cx=q.popleft()
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and (ny,nx) not in reg and a[ny,nx] in (2,color):
                        reg.add((ny,nx)); q.append((ny,nx))
        rys=[p[0] for p in reg]; rxs=[p[1] for p in reg]
        patch=a[min(rys):max(rys)+1,min(rxs):max(rxs)+1]
        if patch.shape!=(3,3):
            pp=np.full((3,3),2,int); pp[:patch.shape[0],:patch.shape[1]]=patch; patch=pp
        res.append(((patch==color).astype(np.int8),color))
    return res
def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield x; yield np.fliplr(x)
def solve(a):
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2; keys=get_keys(a,box); hole=(canvas==0)
    cand=[]  # each: (paints frozenset, color, bgcells frozenset)
    for ty in range(H-2):
        for tx in range(W-2):
            win=hole[ty:ty+3,tx:tx+3]
            if not win.any(): continue
            for fg,color in keys:
                for ff in orients(fg):
                    if np.array_equal((ff==0),win):
                        paints=frozenset((ty+dr,tx+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==1)
                        bgc=frozenset((ty+dr,tx+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==0)
                        cand.append((paints,color,bgc))
    # need: cover every hole cell exactly once (each hole belongs to one key bg), paints non-overlapping,
    # paints must not cover hole cells. Select subset of cand s.t. union(bgc)=all holes, bgc disjoint, paints disjoint.
    allholes=frozenset(zip(*np.where(hole))) if hole.any() else frozenset()
    allholes=frozenset((int(y),int(x)) for y,x in zip(*np.where(hole)))
    cand=list({c for c in cand})
    # backtrack
    # group candidates by which holes they cover
    chosen=[]
    def bt(remaining, usedpaint, idx_used):
        if not remaining: return True
        # pick a hole, find candidates covering it
        h=next(iter(remaining))
        opts=[c for c in cand if h in c[2] and c[2]<=remaining and not (c[0]&usedpaint) and not (c[0]&allholes)]
        for c in opts:
            chosen.append(c)
            if bt(remaining-c[2], usedpaint|c[0], None): return True
            chosen.pop()
        return False
    if allholes:
        bt(allholes, frozenset(), None)
    for paints,color,bgc in chosen:
        for (yy,xx) in paints: out[yy,xx]=color
    return out
ok=True
for split in ['train','test','arc-gen']:
    cnt=0;bad=0;badidx=[]
    for idx,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        try: p=solve(a)
        except Exception as e: bad+=1;badidx.append((idx,str(e)[:25]));ok=False;cnt+=1;continue
        if not (p.shape==b.shape and np.array_equal(p,b)): bad+=1;badidx.append(idx);ok=False
        cnt+=1
    print(split,'total',cnt,'bad',bad,'first',badidx[:8])
print('ALL',ok)
