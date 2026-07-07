import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
from collections import deque,Counter
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
    lbl,n=ndimage.label(nz,structure=S8); keys=[]; H,W=a.shape
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        patch=a[y0:y1+1,x0:x1+1]; cols=set(patch.flatten().tolist())-{0,2}
        if len(cols)!=1: continue
        color=cols.pop()
        fcells=[(yy,xx) for yy in range(y0,y1+1) for xx in range(x0,x1+1) if a[yy,xx]==color]
        start=fcells[0]; q=deque([start]); reg=set([start])
        while q:
            cy,cx=q.popleft()
            for dy in(-1,0,1):
                for dx in(-1,0,1):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and (ny,nx) not in reg and a[ny,nx] in (2,color):
                        reg.add((ny,nx)); q.append((ny,nx))
        rys=[p[0] for p in reg]; rxs=[p[1] for p in reg]
        patch=a[min(rys):max(rys)+1,min(rxs):max(rxs)+1]
        if patch.shape!=(3,3):
            pp=np.full((3,3),2,int); pp[:patch.shape[0],:patch.shape[1]]=patch; patch=pp
        keys.append(((patch==color).astype(np.int8),int(color)))
    return keys
def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield('r%d'%k,x); yield('r%df'%k,np.fliplr(x))
idx=3
a=np.array(d['arc-gen'][idx]['input']);b=np.array(d['arc-gen'][idx]['output'])
box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy();H,W=canvas.shape
keys=get_keys(a,box); hole=(canvas==0)
print('keys:',[(k[1],k[0].tolist()) for k in keys])
print('canvas:')
for r in canvas:print(''.join(str(x) for x in r))
print('expected out:')
for r in b:print(''.join(str(x) for x in r))
# windows with multiple matches
for ty in range(H-2):
    for tx in range(W-2):
        win=hole[ty:ty+3,tx:tx+3]
        if not win.any():continue
        ms=[]
        for fg,color in keys:
            for nm,ff in orients(fg):
                if np.array_equal((ff==0),win): ms.append((nm,color,ff))
        if len(ms)>1:
            print('window ty',ty,'tx',tx,'matches:')
            for nm,color,ff in ms:
                paints=[(ty+dr,tx+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==1]
                ok=all(b[yy,xx]==color for yy,xx in paints)
                print('  ',nm,'color',color,'fg',ff.tolist(),'EXP' if ok else 'wrong')
