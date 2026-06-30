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
    # rank: identity-ish first
    order=[(0,False),(0,True),(1,False),(3,False),(1,True),(3,True),(2,False),(2,True)]
    for k,fl in order:
        x=np.rot90(m,k)
        if fl: x=np.fliplr(x)
        yield x
def solve(a):
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2; keys=get_keys(a,box)
    PAD=2; cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
    hl,hn=ndimage.label(canvas==0,structure=S8)
    for i in range(1,hn+1):
        ys,xs=np.where(hl==i); cells=set(zip((ys+PAD).tolist(),(xs+PAD).tolist()))
        ymin,ymax=ys.min()+PAD,ys.max()+PAD; xmin,xmax=xs.min()+PAD,xs.max()+PAD
        chosen=None
        for ty in range(ymin,ymax-1):  # prefer top-left-most window first? iterate ty ascending
            pass
        # iterate placements top-left first, orientations in ranked order, take first valid
        done=False
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
                            for dr in range(3):
                                for dc in range(3):
                                    yy=ty+dr-PAD;xx=tx+dc-PAD
                                    if 0<=yy<H and 0<=xx<W and ff[dr,dc]==1: out[yy,xx]=color
                            done=True;break
                    if done:break
                if done:break
            if done:break
    return out
ok=True
for split in ['train','test','arc-gen']:
    cnt=0;bad=0;badidx=[]
    for idx,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        try: p=solve(a)
        except Exception as e: bad+=1;badidx.append((idx,str(e)[:20]));ok=False;cnt+=1;continue
        if not (p.shape==b.shape and np.array_equal(p,b)): bad+=1;badidx.append(idx);ok=False
        cnt+=1
    print(split,'total',cnt,'bad',bad,'first',badidx[:8])
print('ALL',ok)
