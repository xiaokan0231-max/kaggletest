import json, numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    d = json.load(f)

for i, ex in enumerate(d['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    box_color = 2
    mask_box = (inp == box_color)
    obj_color = [c for c in np.unique(inp) if c != 0 and c != box_color][0]
    mask_obj = (inp == obj_color)
    
    # Crop obj
    r_obj, c_obj = np.where(mask_obj)
    o_rmin, o_rmax = np.min(r_obj), np.max(r_obj)
    o_cmin, o_cmax = np.min(c_obj), np.max(c_obj)
    obj_crop = inp[o_rmin:o_rmax+1, o_cmin:o_cmax+1]
    
    # Scale obj by 2x
    obj_scaled = np.repeat(np.repeat(obj_crop, 2, axis=0), 2, axis=1)
    
    # Crop box
    r_box, c_box = np.where(mask_box)
    b_rmin, b_rmax = np.min(r_box), np.max(r_box)
    b_cmin, b_cmax = np.min(c_box), np.max(c_box)
    box_crop = inp[b_rmin:b_rmax+1, b_cmin:b_cmax+1]
    
    # Place scaled obj in center of box
    result = np.copy(box_crop)
    H_b, W_b = box_crop.shape
    H_o, W_o = obj_scaled.shape
    r_start = (H_b - H_o) // 2
    c_start = (W_b - W_o) // 2
    
    # Create mask for where object goes
    for r in range(H_o):
        for c in range(W_o):
            if obj_scaled[r, c] != 0:
                result[r_start+r, c_start+c] = obj_scaled[r, c]
                
    if np.array_equal(result, out):
        print(f"Example {i} Match!")
    else:
        print(f"Example {i} Failed!")
