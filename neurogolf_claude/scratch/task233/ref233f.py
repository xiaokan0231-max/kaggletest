import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
d=ng.load_examples(233)

def find_canvas(a):
    mask=(a==2)
    lbl,n=ndimage.label(mask)
    best=None;bs=-1
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i)
        h=ys.max()-ys.min()+1;w=xs.max()-xs.min()+1
        if h*w>bs: bs=h*w;best=(ys.min(),ys.max(),xs.min(),xs.max())
    return best

def get_keys(a,box):
    # A key is a 3x3 block (outside canvas) containing exactly one fg color (!=0,!=2) plus 2-background,
    # and it is isolated (its 3x3 bbox of non-zero stuff). Detect by: connected components of (a!=0) outside canvas.
    r0,r1,c0,c1=box
    nonzero=(a!=0)
    nonzero[r0:r1+1,c0:c1+1]=False
    lbl,n=ndimage.label(nonzero)  # 4-connectivity; 3x3 block of 2s+fg is connected (2s connect)
    keys=[]
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i)
        y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        patch=a[y0:y1+1,x0:x1+1]
        cols=set(patch.flatten().tolist())-{0,2}
        if len(cols)!=1: continue
        color=cols.pop()
        fg=np.full((3,3),0,np.int8)
        fg[:patch.shape[0],:patch.shape[1]]=(patch==color)
        keys.append((fg,int(color)))
    return keys

def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield x; yield np.fliplr(x)

def solve(a):
    box=find_canvas(a);r0,r1,c0,c1=box
    canvas=a[r0:r1+1,c0:c1+1].copy()
    H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2
    keys=get_keys(a,box)
    PAD=2
    cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
    holemask=(canvas==0)
    hl,hn=ndimage.label(holemask)
    for i in range(1,hn+1):
        ys,xs=np.where(hl==i)
        cells=set(zip((ys+PAD).tolist(),(xs+PAD).tolist()))
        ymin,ymax=ys.min()+PAD,ys.max()+PAD; xmin,xmax=xs.min()+PAD,xs.max()+PAD
        cands=[]
        for ty in range(ymax-2,ymin+1):
            for tx in range(xmax-2,xmin+1):
                if not all(ty<=y<ty+3 and tx<=x<tx+3 for y,x in cells): continue
                for fg,color in keys:
                    for ff in orients(fg):
                        bg=(ff==0); good=True;clip=0
                        for dr in range(3):
                            for dc in range(3):
                                yy=ty+dr;xx=tx+dc; val=cv[yy,xx]
                                if val==-1:
                                    if ff[dr,dc]==1: good=False;break
                                    clip+=1
                                else:
                                    if bool(bg[dr,dc])!=(val==0): good=False;break
                            if not good:break
                        if good:
                            # require: every 0-cell in the 3x3 window belongs to THIS blob
                            winholes=set()
                            for dr2 in range(3):
                                for dc2 in range(3):
                                    yy2=ty+dr2;xx2=tx+dc2;v2=cv[yy2,xx2]
                                    if v2==0: winholes.add((yy2,xx2))
                            if winholes==cells:
                                cands.append((clip,ty,tx,ff,color))
        if not cands: continue
        cands.sort(key=lambda t:t[0])
        clip,ty,tx,ff,color=cands[0]
        for dr in range(3):
            for dc in range(3):
                yy=ty+dr-PAD;xx=tx+dc-PAD
                if 0<=yy<H and 0<=xx<W and ff[dr,dc]==1: out[yy,xx]=color
    return out

ok=True
for split in ['train','test','arc-gen']:
    cnt=0;bad=0;badidx=[]
    for idx,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        p=solve(a)
        if not (p.shape==b.shape and np.array_equal(p,b)): bad+=1;badidx.append(idx);ok=False
        cnt+=1
    print(split,'total',cnt,'bad',bad,'first',badidx[:6])
print('ALL',ok)


ok=True
for split in ['train','test','arc-gen']:
    cnt=0;bad=0;badidx=[]
    for idx,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        p=solve(a)
        if not (p.shape==b.shape and np.array_equal(p,b)): bad+=1;badidx.append(idx);ok=False
        cnt+=1
    print(split,'total',cnt,'bad',bad,'first',badidx[:8])
print('ALL',ok)
