import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    print(f'\n--- Train {i} ---')
    
    # find components in output
    for c in range(1, 10):
        if np.any(out == c):
            rows, cols = np.where(out == c)
            r_min, r_max = rows.min(), rows.max()
            c_min, c_max = cols.min(), cols.max()
            print(f'Color {c} shape:')
            print((out[r_min:r_max+1, c_min:c_max+1] == c).astype(int))

