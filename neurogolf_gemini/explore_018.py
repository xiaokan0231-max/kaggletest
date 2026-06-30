import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    print(f"--- Train {i} ---")
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    print("Input non-zero:")
    r, c = np.where(inp != 0)
    for rr, cc in zip(r, c):
        print(f"  ({rr}, {cc}): {inp[rr, cc]}")
        
    print("Output non-zero:")
    r, c = np.where(out != 0)
    for rr, cc in zip(r, c):
        print(f"  ({rr}, {cc}): {out[rr, cc]}")

    print("Input shape:", inp.shape)
    
