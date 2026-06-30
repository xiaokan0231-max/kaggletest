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
        color=cols.pop(); fg=np.zeros((3,3),np.int8); fg[:patch.shape[0],:patch.shape[1]]=(patch==color)
        keys.append((fg,int(color)))
    return keys
def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield x; yield np.fliplr(x)
def solve(a, b):
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2; keys=get_keys(a,box)
    PAD=2; cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
    hl,hn=ndimage.label(canvas==0,structure=S8)
    if b is None or b.shape!=(H,W): return None,0,0
    ambig=0; total=0
    for i in range(1,hn+1):
        ys,xs=np.where(hl==i); cells=set(zip((ys+PAD).tolist(),(xs+PAD).tolist()))
        ymin,ymax=ys.min()+PAD,ys.max()+PAD; xmin,xmax=xs.min()+PAD,xs.max()+PAD
        valid=[]
        for ty in range(ymax-2,ymin+1):
            for tx in range(xmax-2,xmin+1):
                if not all(ty<=y<ty+3 and tx<=x<tx+3 for y,x in cells): continue
                winholes=set((ty+dr,tx+dc) for dr in range(3) for dc in range(3) if cv[ty+dr,tx+dc]==0)
                if winholes!=cells: continue
                for fg,color in keys:
                    for ff in orients(fg):
                        bg=(ff==0); good=True;fgout=False
                        for dr in range(3):
                            for dc in range(3):
                                yy=ty+dr;xx=tx+dc;val=cv[yy,xx]
                                if val==-1:
                                    if ff[dr,dc]==1: fgout=True
                                else:
                                    if bool(bg[dr,dc])!=(val==0): good=False
                        if good and not fgout:
                            paints=frozenset((ty+dr-PAD,tx+dc-PAD) for dr in range(3) for dc in range(3) if ff[dr,dc]==1 and 0<=ty+dr-PAD<H and 0<=tx+dc-PAD<W)
                            valid.append((paints,color))
        # distinct paint-results
        distinct=set(valid)
        # which match expected?
        correct=[v for v in distinct if all(b[yy,xx]==v[1] for (yy,xx) in v[0])]
        total+=1
        if len(distinct)>1: ambig+=1
        # apply a correct one if exists
        chosen = correct[0] if correct else (list(distinct)[0] if distinct else None)
        if chosen:
            for (yy,xx) in chosen[0]: out[yy,xx]=chosen[1]
    return out,ambig,total
TA=0;TT=0
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        a=np.array(ex['input']);b=np.array(ex['output'])
        _,am,tt=solve(a,b); TA+=am;TT+=tt
print('blobs with >1 distinct paint-result (ambiguous):',TA,'of',TT)
