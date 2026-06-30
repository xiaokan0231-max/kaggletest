import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

out = np.array(data['train'][1]['output'])

print('Train 1 Output:')
for row in out:
    print(' '.join(str(x) if x != 0 else '.' for x in row))

