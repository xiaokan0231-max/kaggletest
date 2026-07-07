import json
import numpy as np

with open('neurogolf/data/raw/task076.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    print(f'\n--- Train {i} ---')
    
    # Find components of 4
    from scipy.ndimage import label
    labeled, num_features = label(inp == 4)
    
    for c in range(1, num_features + 1):
        mask = (labeled == c)
        rows, cols = np.where(mask)
        r_min, r_max = rows.min(), rows.max()
        c_min, c_max = cols.min(), cols.max()
        
        # pad by 2 to see the surroundings
        r_min = max(0, r_min - 2)
        r_max = min(inp.shape[0] - 1, r_max + 2)
        c_min = max(0, c_min - 2)
        c_max = min(inp.shape[1] - 1, c_max + 2)
        
        print(f'\nComponent {c}:')
        print('Input:')
        print(inp[r_min:r_max+1, c_min:c_max+1])
        print('Output:')
        print(out[r_min:r_max+1, c_min:c_max+1])
