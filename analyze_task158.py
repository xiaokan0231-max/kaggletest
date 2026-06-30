import json
import numpy as np

with open('neurogolf/data/raw/task158.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    print(f'\n--- Train {i} ---')
    print('Input shape:', inp.shape)
    print('Output shape:', out.shape)
    diff = (inp != out)
    if np.any(diff):
        print('Changed FROM:', np.unique(inp[diff]))
        print('Changed TO:', np.unique(out[diff]))
        rows, cols = np.where(diff)
        r_min, r_max = max(0, min(rows)-1), min(inp.shape[0]-1, max(rows)+1)
        c_min, c_max = max(0, min(cols)-1), min(inp.shape[1]-1, max(cols)+1)
        print('Input snippet around changes:')
        print(inp[r_min:r_max+1, c_min:c_max+1])
        print('Output snippet around changes:')
        print(out[r_min:r_max+1, c_min:c_max+1])
