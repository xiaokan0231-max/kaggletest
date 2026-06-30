import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

inp = np.array(data['train'][0]['input'])

for c in [7, 2, 1, 4]:
    r, cols = np.where(inp == c)
    for pr, pc in zip(r, cols):
        if pr >= 7 and pr <= 13 and pc >= 12 and pc <= 18:
            print(f"Color {c}: ({pr}, {pc})")

