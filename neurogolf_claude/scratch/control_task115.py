import json
import numpy as np
# Control: a WRONG solver (always row, never collapse) must score <1.0, proving the harness can fail.
def solve_bad(i):
    a=np.array(i); H,W=a.shape
    rs=[int(np.bincount(a[r,:]).argmax()) for r in range(H)]
    return np.array(rs).reshape(-1,1)
d=json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task115.json'))
tot=ok=0
for split in ['train','test','arc-gen']:
    for ex in d[split]:
        tot+=1
        p=solve_bad(ex['input']); e=np.array(ex['output'])
        if p.shape==e.shape and np.array_equal(p,e): ok+=1
print(f"BAD control: {ok}/{tot} = {ok/tot:.4f}  (must be <<1.0 to prove harness detects failures)")
