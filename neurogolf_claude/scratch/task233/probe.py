import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
d=ng.load_examples(233)
def find_canvas(a):
    mask=(a==2); lbl,n=ndimage.label(mask); best=None;bs=-1
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i); h=ys.max()-ys.min()+1;w=xs.max()-xs.min()+1
        if h*w>bs: bs=h*w;best=(ys.min(),ys.max(),xs.min(),xs.max())
    return best
def get_keys(a,box):
    r0,r1,c0,c1=box; nz=(a!=0); nz[r0:r1+1,c0:c1+1]=False
    lbl,n=ndimage.label(nz); keys=[]
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
# For train0 and train1, for each hole blob, list valid candidates and which paint matches expected
for ti in [0,1]:
    a=np.array(d['train'][ti]['input']);b=np.array(d['train'][ti]['output'])
    box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1]; H,W=canvas.shape
    keys=get_keys(a,box)
    PAD=2; cv=np.full((H+2*PAD,W+2*PAD),-1,int); cv[PAD:PAD+H,PAD:PAD+W]=canvas
    hl,hn=ndimage.label(canvas==0)
    print('=== train',ti,'keys colors',[k[1] for k in keys])
    for i in range(1,hn+1):
        ys,xs=np.where(hl==i); cells=set(zip((ys+PAD).tolist(),(xs+PAD).tolist()))
        ymin,ymax=ys.min()+PAD,ys.max()+PAD; xmin,xmax=xs.min()+PAD,xs.max()+PAD
        print(' blob',i,'cells(canvas)',sorted((y-PAD,x-PAD) for y,x in cells))
        for ty in range(ymax-2,ymin+1):
            for tx in range(xmax-2,xmin+1):
                if not all(ty<=y<ty+3 and tx<=x<tx+3 for y,x in cells): continue
                for fg,color in keys:
                    for nm,ff in orients(fg):
                        bg=(ff==0); good=True;clip=0;fgout=False
                        for dr in range(3):
                            for dc in range(3):
                                yy=ty+dr;xx=tx+dc;val=cv[yy,xx]
                                if val==-1:
                                    if ff[dr,dc]==1: fgout=True
                                    clip+=1
                                else:
                                    if bool(bg[dr,dc])!=(val==0): good=False
                        if good and not fgout:
                            # paint & check vs expected
                            o=canvas.copy(); 
                            painted=[]
                            for dr in range(3):
                                for dc in range(3):
                                    yy=ty+dr-PAD;xx=tx+dc-PAD
                                    if 0<=yy<H and 0<=xx<W and ff[dr,dc]==1: painted.append((yy,xx))
                            okexp=all(b[yy,xx]==color for yy,xx in painted)
                            print('   ty',ty-PAD,'tx',tx-PAD,nm,'color',color,'clip',clip,'nfg',len(painted),'EXPOK' if okexp else 'WRONG')
