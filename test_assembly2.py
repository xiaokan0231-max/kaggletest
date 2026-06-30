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
    
    rows_obj = np.any(mask_obj, axis=1)
    cols_obj = np.any(mask_obj, axis=0)
    o_rmin, o_rmax = np.where(rows_obj)[0][[0, -1]]
    o_cmin, o_cmax = np.where(cols_obj)[0][[0, -1]]
    
    # Try top-left
    dy1 = b_rmin - o_rmin
    dx1 = b_cmin - o_cmin
    
    # Try center
    dy2 = (b_rmin + b_rmax) // 2 - (o_rmin + o_rmax) // 2
    dx2 = (b_cmin + b_cmax) // 2 - (o_cmin + o_cmax) // 2
    
    # Check which dy, dx works
    for name, dy, dx in [("TopLeft", dy1, dx1), ("Center", dy2, dx2)]:
        obj_translated = np.roll(inp * mask_obj, shift=(dy, dx), axis=(0, 1))
        merged = np.maximum(inp * mask_box, obj_translated)
        crop = merged[b_rmin:b_rmax+1, b_cmin:b_cmax+1]
        if np.array_equal(crop, out):
            print(f"Example {i} Match! {name}")
            break
    else:
        print(f"Example {i} Failed!")
