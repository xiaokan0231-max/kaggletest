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
        x=np.rot90(m,k); yield ('r%d'%k,x); yield ('r%df'%k,np.fliplr(x))
shown=0
for split in ['train','test','arc-gen']:
    for ei,ex in enumerate(d[split]):
        if shown>=6: break
        a=np.array(ex['input']);b=np.array(ex['output'])
        box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
        if b.shape!=(H,W): continue
        keys=get_keys(a,box)
        PAD=2; cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
        hl,hn=ndimage.label(canvas==0,structure=S8)
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
                        for nm,ff in orients(fg):
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
                                valid.append((ty-PAD,tx-PAD,nm,color,paints))
            distinct={(v[4],v[3]) for v in valid}
            if len(distinct)>1:
                shown+=1
                print('=== %s%d blob cells(canvas):'%(split,ei),sorted((y-PAD,x-PAD) for y,x in cells))
                for v in valid:
                    correct=all(b[yy,xx]==v[3] for (yy,xx) in v[4])
                    print('   ty%d tx%d %s color%d paints%s %s'%(v[0],v[1],v[2],v[3],sorted(v[4]),'CORRECT' if correct else 'wrong'))
                if shown>=6: break
        if shown>=6: break
    if shown>=6: break
