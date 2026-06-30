import json
import numpy as np

with open('neurogolf/data/raw/task158.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    print(f'\n--- Train {i} ---')
    print('Input:')
    print(inp)
    print('Output:')
    print(out)
