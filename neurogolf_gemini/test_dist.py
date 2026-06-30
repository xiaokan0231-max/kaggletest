import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

for ex_idx, ex in enumerate(data['train'] + data['test']):
    inp = np.array(ex['input'])
    
    unique, counts = np.unique(inp, return_counts=True)
    mask = unique != 0
    unique, counts = unique[mask], counts[mask]
    sorted_indices = np.argsort(counts)
    anchors = unique[sorted_indices[:3]]
    struct = unique[sorted_indices[-1]]
    
    print(f"Ex {ex_idx}: shape {inp.shape}")
    r_s, c_s = np.where(inp == struct)
    for a in anchors:
        r_a, c_a = np.where(inp == a)
        # For each anchor pixel, find distance to nearest structure pixel
        for rr, cc in zip(r_a, c_a):
            if len(r_s) > 0:
                dist = np.max(np.abs(r_s - rr) + np.abs(c_s - cc)) # actually min dist
                min_dist = np.min(np.maximum(np.abs(r_s - rr), np.abs(c_s - cc)))
                print(f"  Anchor {a} at ({rr},{cc}) dist to struct: {min_dist}")
            
