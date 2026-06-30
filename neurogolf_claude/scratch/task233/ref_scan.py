import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
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
    lbl,n=ndimage.label(nz,structure=S8); keys=[]
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        patch=a[y0:y1+1,x0:x1+1]; cols=set(patch.flatten().tolist())-{0,2}
        if len(cols)!=1: continue
        color=cols.pop()
        # key is 3x3: but fg may not fill bbox. Reconstruct 3x3 from the surrounding context:
        # The key occupies a 3x3 with 2-background and 0 outside. Find the 3x3 block = bbox of the
        # connected (2 or fg) region around it. Simpler: expand bbox to include adjacent 2s forming 3x3.
        # Take the bbox of cells that are (fg or 2) connected to this fg, limited to 3x3.
        keys.append((color,(y0,y1,x0,x1)))
    # reconstruct full 3x3 key by looking at the (2|fg) block
    out=[]
    seen=set()
    for color,(y0,y1,x0,x1) in keys:
        # the 3x3 block: the key region is exactly 3x3 of (2|color) surrounded by 0. find it.
        # scan: the block containing (y0,x0)
        # grow region of value in {2,color} 8-connected
        from collections import deque
        H,W=a.shape
        # start from a fg cell
        start=(y0,x0)
        # find a fg cell
        fcells=[(yy,xx) for yy in range(y0,y1+1) for xx in range(x0,x1+1) if a[yy,xx]==color]
        start=fcells[0]
        q=deque([start]); reg=set([start])
        while q:
            cy,cx=q.popleft()
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and (ny,nx) not in reg and a[ny,nx] in (2,color):
                        reg.add((ny,nx)); q.append((ny,nx))
        rys=[p[0] for p in reg]; rxs=[p[1] for p in reg]
        ky0,ky1,kx0,kx1=min(rys),max(rys),min(rxs),max(rxs)
        patch=a[ky0:ky1+1,kx0:kx1+1]
        if patch.shape!=(3,3): 
            # pad to 3x3 (rare)
            pp=np.full((3,3),2,int); pp[:patch.shape[0],:patch.shape[1]]=patch; patch=pp
        fg=(patch==color).astype(np.int8)
        out.append((fg,int(color)))
    return out
def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield x; yield np.fliplr(x)
def solve(a):
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2; keys=get_keys(a,box)
    hole=(canvas==0)
    # window scan: for each top-left (ty,tx) with full 3x3 inside canvas, check each key orientation:
    # bg(==0)==hole-window AND all those holes covered. Collect placements.
    placements=[]
    for ty in range(H-2):
        for tx in range(W-2):
            win=hole[ty:ty+3,tx:tx+3]
            if not win.any(): continue
            for fg,color in keys:
                for ff in orients(fg):
                    bg=(ff==0)
                    if np.array_equal(bg,win):
                        placements.append((ty,tx,ff,color))
    # paint all (assume consistent / non-overlapping). Apply.
    for ty,tx,ff,color in placements:
        for dr in range(3):
            for dc in range(3):
                if ff[dr,dc]==1: out[ty+dr,tx+dc]=color
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
