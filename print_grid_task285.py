import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

inp = np.array(data['train'][0]['input'])
out = np.array(data['train'][0]['output'])

print('Output:')
for row in out:
    print(' '.join(str(x) if x != 0 else '.' for x in row))

