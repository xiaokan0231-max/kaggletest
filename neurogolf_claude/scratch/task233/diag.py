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
a=np.array(d['arc-gen'][1]['input']); b=np.array(d['arc-gen'][1]['output'])
print('IN',a.shape)
for ri,r in enumerate(a): print(f'{ri:2d}',''.join(str(x) for x in r))
print('OUT',b.shape)
for ri,r in enumerate(b): print(f'{ri:2d}',''.join(str(x) for x in r))
box=find_canvas(a); print('canvas box',box,'-> shape',(box[1]-box[0]+1,box[3]-box[2]+1))
