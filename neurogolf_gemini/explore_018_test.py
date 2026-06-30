import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['test']):
    print(f"--- Test {i} ---")
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    print("Input unique:", np.unique(inp))
    for c in np.unique(inp):
        if c == 0: continue
        r, cc = np.where(inp == c)
        print(f"Color {c}: {len(r)} pixels")
    for c in [1, 2, 4, 5]:
        r, cc = np.where(inp == c)
        for rr, ccc in zip(r, cc):
            print(f"  Color {c} at ({rr}, {ccc})")
