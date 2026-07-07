import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

out = np.array(data['train'][0]['output'])

for r in range(6, 16):
    print(f"Row {r:2d}: " + " ".join(str(x) if x != 0 else '.' for x in out[r, 13:20]))

