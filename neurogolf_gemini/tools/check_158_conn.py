import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'])
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    print(f"Ex {i}")
    for c in non_bg:
        mask = (inp == c)
        labeled, num_features = ndimage.label(mask)
        print(f"  Color {c}: {num_features} components")
