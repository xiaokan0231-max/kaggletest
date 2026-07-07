import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

s = ndimage.generate_binary_structure(2, 2)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'])
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    conn_color = None
    for c in non_bg:
        if ndimage.label(inp == c, structure=s)[1] == 1:
            conn_color = c
            break
            
    endpoints = [c for c in non_bg if c != conn_color]
    conn_mask = (inp == conn_color)
    conn_dilated = ndimage.binary_dilation(conn_mask, structure=s)
    
    print(f"Ex {i}")
    for c in endpoints:
        sizes = []
        labeled, n = ndimage.label(inp == c, structure=s)
        for comp in range(1, n+1):
            mask = (labeled == comp)
            if not np.any(mask & conn_dilated):
                sizes.append(mask.sum())
        print(f"  Endpoint {c} target sizes: {sorted(sizes)}")
