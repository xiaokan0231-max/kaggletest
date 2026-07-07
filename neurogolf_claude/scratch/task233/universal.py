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
        x=np.rot90(m,k); yield x; yield np.fliplr(x)
# for each example, for each painted color region, find the placement (ty,tx,orient,key) that reproduces the paint,
# then record (holewin, keyfg) relationship
rel=set()
fails=0
for split in ['train','arc-gen']:
    for ei,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy();H,W=canvas.shape
        if b.shape!=(H,W): continue
        keys=get_keys(a,box); hole=(canvas==0)
        paintmask=(b!=2)&(canvas!=0)  # painted (output!=2 and not originally hole)... but hole->2 in out
        # Actually painted cells = output cells that are not 2. (holes become 2)
        paintmask=(b!=2)
        # group painted cells
        lbl,n=ndimage.label(paintmask,structure=S8)
        for i in range(1,n+1):
            pys,pxs=np.where(lbl==i); color=int(b[pys[0],pxs[0]])
            # find 3x3 window + orient + key reproducing exactly these paints AND consistent
            found=False
            y0,y1,x0,x1=pys.min(),pys.max(),pxs.min(),pxs.max()
            for ty in range(max(0,y1-2),min(y0+1,H-2)):
                for tx in range(max(0,x1-2),min(x0+1,W-2)):
                    for fg,kc in keys:
                        if kc!=color: continue
                        for ff in orients(fg):
                            pc=set((ty+dr,tx+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==1)
                            if pc==set(zip(pys.tolist(),pxs.tolist())):
                                hw=hole[ty:ty+3,tx:tx+3].astype(int)
                                rel.add((tuple(ff.flatten()),tuple(hw.flatten())))
                                found=True;break
                        if found:break
                    if found:break
                if found:break
            if not found: fails+=1
print('relations (fg,holewin) count',len(rel),'unmatched paint-regions',fails)
# check: is holewin always == fg? or == NOT fg? or fg subset hole?
eq=0;comp=0;sub=0;other=0
for fg,hw in rel:
    fg=np.array(fg);hw=np.array(hw)
    if np.array_equal(fg,hw): eq+=1
    elif np.array_equal(1-fg,hw): comp+=1
    elif np.all(hw[fg==1]==1): sub+=1
    else: other+=1
print('fg==hole',eq,'hole==NOTfg',comp,'fg subset hole',sub,'other',other)
