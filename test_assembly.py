import json, numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    d = json.load(f)
    
for ex in d['train']:
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    box_color = 2
    mask_box = (inp == box_color)
    mask_obj = (inp != 0) & (inp != box_color)
    
    rows_box = np.any(mask_box, axis=1)
    cols_box = np.any(mask_box, axis=0)
    b_rmin, b_rmax = np.where(rows_box)[0][[0, -1]]
    b_cmin, b_cmax = np.where(cols_box)[0][[0, -1]]
    
    rows_obj = np.any(mask_obj, axis=1)
    cols_obj = np.any(mask_obj, axis=0)
    o_rmin, o_rmax = np.where(rows_obj)[0][[0, -1]]
    o_cmin, o_cmax = np.where(cols_obj)[0][[0, -1]]
    
    # Try center alignment
    b_rcenter = (b_rmin + b_rmax) // 2
    b_ccenter = (b_cmin + b_cmax) // 2
    o_rcenter = (o_rmin + o_rmax) // 2
    o_ccenter = (o_cmin + o_cmax) // 2
    
    dy = b_rcenter - o_rcenter
    dx = b_ccenter - o_ccenter
    
    # Translate object
    obj_translated = np.roll(inp * mask_obj, shift=(dy, dx), axis=(0, 1))
    
    # Merge with box
    merged = np.maximum(inp * mask_box, obj_translated)
    
    # Crop box
    crop = merged[b_rmin:b_rmax+1, b_cmin:b_cmax+1]
    
    print(np.array_equal(crop, out))
