import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
from collections import deque
d=ng.load_examples(233)
S8=np.ones((3,3),int)
exec(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task233/universal.py').read().split('def orients')[0].split('# for each')[0].split('d=ng.load')[1] if False else open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task233/universal.py').read().split('def orients(m):')[0])
def orients(m):
    for k in range(4):
        x=np.rot90(m,k); yield x; yield np.fliplr(x)
# Hypothesis: paint cells = key-fg (some D4), and EVERY painted cell was a hole, and the hole region
# = key-fg(transformed) UNION key-bg-cells-that-are-holes... 
# Let's test: for each painted region, does paint-mask match some D4 of key-fg, with all paint cells being holes?
stats={'paint==fgD4_onholes':0,'no':0}
examples_no=[]
for split in ['train','arc-gen']:
    for ei,ex in enumerate(d[split]):
        a=np.array(ex['input']);b=np.array(ex['output'])
        box=find_canvas(a);r0,r1,c0,c1=box; canvas=a[r0:r1+1,c0:c1+1].copy();H,W=canvas.shape
        if b.shape!=(H,W): continue
        keys=get_keys(a,box); hole=(canvas==0)
        pm=(b!=2)
        lbl,n=ndimage.label(pm,structure=S8)
        for i in range(1,n+1):
            pys,pxs=np.where(lbl==i); color=int(b[pys[0],pxs[0]])
            pcells=set(zip(pys.tolist(),pxs.tolist()))
            # all painted cells must be holes in canvas?
            allhole=all(canvas[y,x]==0 for y,x in pcells)
            kf=[fg for fg,kc in keys if kc==color]
            matched=False
            if kf:
                fg=kf[0]
                y0,x0=pys.min(),pxs.min()
                for ff in orients(fg):
                    fcells=set((y0+dr,x0+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==1)
                    # normalize fcells to top-left
                    if not fcells: continue
                    fy=min(c[0] for c in fcells); fx=min(c[1] for c in fcells)
                    fnorm=set((c[0]-fy,c[1]-fx) for c in fcells)
                    py=min(c[0] for c in pcells); px=min(c[1] for c in pcells)
                    pnorm=set((c[0]-py,c[1]-px) for c in pcells)
                    if fnorm==pnorm: matched=True;break
            if matched and allhole: stats['paint==fgD4_onholes']+=1
            else:
                stats['no']+=1
                if len(examples_no)<5: examples_no.append((split,ei,color,'allhole',allhole,'fgmatch',matched))
print(stats)
for e in examples_no: print(e)
