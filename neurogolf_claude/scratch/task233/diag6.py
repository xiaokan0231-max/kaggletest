import sys; sys.path.insert(0,'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils as ng; ng._NEUROGOLF_DIR='/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/'
import numpy as np
from scipy import ndimage
d=ng.load_examples(233)
a=np.array(d['arc-gen'][21]['input']);b=np.array(d['arc-gen'][21]['output'])
print('full input',a.shape)
for ri,r in enumerate(a):print(f'{ri:2d}',''.join(str(x) for x in r))
print('full output',b.shape)
for ri,r in enumerate(b):print(f'{ri:2d}',''.join(str(x) for x in r))
