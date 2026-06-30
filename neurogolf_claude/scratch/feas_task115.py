import json
import numpy as np
d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task115.json'))
# Output is VARIABLE length: number of bands varies per example. Tabulate output lengths.
from collections import Counter
lens=Counter()
oriented=Counter()
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        exp=np.array(ex['output'])
        lens[max(exp.shape)]+=1
        oriented[exp.shape if 1 in exp.shape else 'multi']  # noqa
print("distinct output band-counts:", sorted(lens.items()))
# input sizes
isz=Counter()
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        isz[np.array(ex['input']).shape]+=1
print("num distinct input shapes:", len(isz))
print("sample input shapes:", sorted(isz.items())[:10])
