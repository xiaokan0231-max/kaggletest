import json, numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    d = json.load(f)
    
for i, ex in enumerate(d['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    box_color = 2
    mask_box = (inp == box_color)
    mask_obj = (inp != 0) & (inp != box_color)
    
    rows_box = np.any(mask_box, axis=1)
    cols_box = np.any(mask_box, axis=0)
    b_rmin, b_rmax = np.where(rows_box)[0][[0, -1]]
    b_cmin, b_cmax = np.where(cols_box)[0][[0, -1]]
    
    found = False
    for dy in range(-30, 30):
        for dx in range(-30, 30):
            obj_translated = np.roll(inp * mask_obj, shift=(dy, dx), axis=(0, 1))
            merged = np.maximum(inp * mask_box, obj_translated)
            crop = merged[b_rmin:b_rmax+1, b_cmin:b_cmax+1]
            if np.array_equal(crop, out):
                print(f"Example {i} Match! dy={dy}, dx={dx}")
                found = True
                break
        if found: break
