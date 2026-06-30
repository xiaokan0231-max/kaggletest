import json
import numpy as np
from scipy.ndimage import label

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

for split in ['train', 'test']:
    for i, ex in enumerate(data[split]):
        inp = np.array(ex['input'])
        counts = np.bincount(inp.flatten())
        bg = np.argmax(counts)
        for c in range(10):
            if c == bg: continue
            mask = (inp == c)
            labeled, num = label(mask, structure=np.ones((3,3)))
            # count how many components of color c have size > 1
            bases = 0
            for j in range(1, num+1):
                if (labeled == j).sum() > 1:
                    bases += 1
            if bases > 1:
                print(f"{split} {i} color {c} has {bases} bases!")
print("Done checking.")
