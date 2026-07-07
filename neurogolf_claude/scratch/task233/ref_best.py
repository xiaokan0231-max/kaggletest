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
def candidates(canvas,cv,PAD,cells,keys,H,W):
    ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
    ymin,ymax,xmin,xmax=min(ys),max(ys),min(xs),max(xs)
    res=[]
    for ty in range(ymax-2,ymin+1):
        for tx in range(xmax-2,xmin+1):
            if not all(ty<=y<ty+3 and tx<=x<tx+3 for y,x in cells): continue
            winholes=set((ty+dr,tx+dc) for dr in range(3) for dc in range(3) if cv[ty+dr,tx+dc]==0)
            if winholes!=set(cells): continue
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
                        res.append((color,paints))
    return list(set(res))
def solve(a):
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy(); H,W=canvas.shape
    out=canvas.copy(); out[out==0]=2; keys=get_keys(a,box)
    PAD=2; cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
    hl,hn=ndimage.label(canvas==0,structure=S8)
    blobs=[]
    for i in range(1,hn+1):
        ys,xs=np.where(hl==i); cells=frozenset(zip((ys+PAD).tolist(),(xs+PAD).tolist()))
        cand=candidates(canvas,cv,PAD,cells,keys,H,W)
        blobs.append((cells,cand))
    # global: assign so each key-color used count matches key multiplicity? Use backtracking with non-overlap of painted cells
    keycount={}
    for fg,color in keys: keycount[color]=keycount.get(color,0)+1
    order=sorted(range(len(blobs)), key=lambda i:len(blobs[i][1]))  # most-constrained first
    assign=[None]*len(blobs)
    used=set()
    usecnt={}
    def bt(k):
        if k==len(order): return True
        bi=order[k]; cells,cand=blobs[bi]
        for color,paints in cand:
            if paints & used: continue
            if usecnt.get(color,0)+1>keycount.get(color,99): continue
            assign[bi]=(color,paints); used.update(paints); usecnt[color]=usecnt.get(color,0)+1
            if bt(k+1): return True
            used.difference_update(paints); usecnt[color]=usecnt.get(color,0)-1; assign[bi]=None
        return False
    bt(0)
    for bi in range(len(blobs)):
        if assign[bi]:
            color,paints=assign[bi]
            for (yy,xx) in paints: out[yy,xx]=color
        elif blobs[bi][1]:
            color,paints=blobs[bi][1][0]
            for (yy,xx) in paints: out[yy,xx]=color
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

print('--- diag arc-gen 1')
a=np.array(d['arc-gen'][1]['input']); b=np.array(d['arc-gen'][1]['output'])
box=find_canvas(a); keys=get_keys(a,box)
print('keys:',[(k[1],k[0].tolist()) for k in keys])
p=solve(a)
for ri in range(p.shape[0]):
    g=''.join(str(x) for x in p[ri]); e=''.join(str(x) for x in b[ri])
    print(f'{ri:2d} {g}  {e}  {"" if g==e else "<<"}')
