import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
from collections import deque
exec(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_claude/scratch/task233/ref_dedup.py').read().split('bad=[]')[0])
d=ng.load_examples(233)
idx=21
a=np.array(d['arc-gen'][idx]['input']);b=np.array(d['arc-gen'][idx]['output'])
box=find_canvas(a);r0,r1,c0,c1=box;canvas=a[r0:r1+1,c0:c1+1].copy();H,W=canvas.shape
keys=get_keys(a,box);hole=(canvas==0)
print('keys',[(k[1],k[0].tolist()) for k in keys])
def orients2(m):
    for k in range(4):
        x=np.rot90(m,k);yield('r%d'%k,x);yield('r%df'%k,np.fliplr(x))
# find windows where DIFFERENT fg compete
for ty in range(H-2):
    for tx in range(W-2):
        win=hole[ty:ty+3,tx:tx+3]
        if not win.any():continue
        fgs=set()
        info=[]
        for fg,color in keys:
            for nm,ff in orients2(fg):
                if np.array_equal((ff==0),win):
                    fgs.add((tuple(ff.flatten()),color)); info.append((nm,color,ff))
        if len(fgs)>1:
            print('CONFLICT window ty',ty,'tx',tx,'win',win.astype(int).tolist())
            for nm,color,ff in info:
                paints=[(ty+dr,tx+dc) for dr in range(3) for dc in range(3) if ff[dr,dc]==1]
                ok=all(b[yy,xx]==color for yy,xx in paints)
                print('  ',nm,'c',color,'fg',ff.tolist(),'EXP' if ok else 'wrong')
p=solve(a)
print('diff cells:',[(int(r),int(c),int(p[r,c]),int(b[r,c])) for r,c in zip(*np.where(p!=b))])

print('=== region rows15-19 cols9-13 canvas')
for ri in range(15,20):
    print(ri,''.join(str(x) for x in canvas[ri,9:14]))
print('expected:')
for ri in range(15,20):
    print(ri,''.join(str(x) for x in b[ri,9:14]))
# key8 and key3 bg masks
import numpy as np
for fg,color in keys:
    print('key',color,'bg',(fg==0).astype(int).tolist())
