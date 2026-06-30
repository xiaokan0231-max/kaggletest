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
    
    print(f"Ex {i}")
    conn_mask = (inp == conn_color)
    conn_dilated = ndimage.binary_dilation(conn_mask, structure=s)
    
    for c in endpoints:
        c_mask = (inp == c)
        labeled, num_features = ndimage.label(c_mask, structure=s)
        for comp in range(1, num_features + 1):
            comp_mask = (labeled == comp)
            if np.any(comp_mask & conn_dilated):
                print(f"  Endpoint {c} in template size: {comp_mask.sum()}")
