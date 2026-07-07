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
    lbl,n=ndimage.label(nz,structure=S8); res=[]; H,W=a.shape
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
    placements=[]
    for ty in range(H-2):
        for tx in range(W-2):
            win=hole[ty:ty+3,tx:tx+3]
            if not win.any(): continue
            for fg,color in keys:
                for ff in orients(fg):
                    if np.array_equal((ff==0),win):
                        placements.append((ty,tx,ff,color))
    for ty,tx,ff,color in placements:
        for dr in range(3):
            for dc in range(3):
                if ff[dr,dc]==1: out[ty+dr,tx+dc]=color
    return out,placements,hole,canvas
# analyze failures
bad=[]
for idx,ex in enumerate(d['arc-gen']):
    a=np.array(ex['input']);b=np.array(ex['output'])
    try: p,pl,hole,canvas=solve(a)
    except Exception as e: bad.append((idx,'ERR'+str(e)[:20])); continue
    if not (p.shape==b.shape and np.array_equal(p,b)):
        # classify: is failure due to >1 orientation matching same window?
        from collections import Counter
        winc=Counter((ty,tx) for ty,tx,ff,color in pl)
        multi=any(v>1 for v in winc.values())
        # count diff cells
        nd=int((p!=b).sum()) if p.shape==b.shape else -1
        bad.append((idx,'multiorient' if multi else 'other', 'diffcells',nd))
print('num bad',len(bad))
for x in bad[:25]: print(x)
