import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
d=ng.load_examples(233)
a=np.array(d['arc-gen'][21]['input']);b=np.array(d['arc-gen'][21]['output'])
canvas=a[9:29,1:19]; H,W=canvas.shape
# for each painted color region in output, find its 3x3 bbox, the hole pattern, and the paint pattern
diff=(b!=2)&(canvas!=0)  # painted cells (output non-2 that weren't holes)
# group painted cells by color and proximity
from scipy import ndimage as ndi
for color in range(1,10):
    if color==2: continue
    m=(b==color)
    if not m.any(): continue
    lbl,n=ndi.label(m,structure=np.ones((3,3)))
    for i in range(1,n+1):
        ys,xs=np.where(lbl==i)
        y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
        # expand to 3x3 region
        cy=(y0+y1)//2; cx=(x0+x1)//2
        ry0=max(0,min(y0,H-3)); rx0=max(0,min(x0,W-3))
        # take 3x3 window covering paints; find associated holes nearby
        print('color',color,'paints',sorted(zip(ys.tolist(),xs.tolist())))
        # window: paints + adjacent holes form 3x3
        # gather holes within distance 2
        holes=[(r,c) for r in range(max(0,y0-2),min(H,y1+3)) for c in range(max(0,x0-2),min(W,x1+3)) if canvas[r,c]==0]
        print('   nearby holes',holes)
