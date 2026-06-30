import json
import numpy as np
from scipy.ndimage import label, find_objects, binary_dilation

def transform(x, k):
    if k >= 4:
        x = np.flip(x, 1)
        k = k - 4
    if k == 1:
        x = np.rot90(x, 1)
    elif k == 2:
        x = np.rot90(x, 2)
    elif k == 3:
        x = np.rot90(x, 3)
    return x

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    print(f'\n--- Train {i} ---')
    
    for c in range(1, 10):
        mask = (inp == c)
        if mask.sum() > 1:
            labeled, num = label(mask)
            for j in range(1, num+1):
                base_mask = (labeled == j)
                slices = find_objects(base_mask.astype(int))[0]
                r_min, r_max = slices[0].start, slices[0].stop - 1
                c_min, c_max = slices[1].start, slices[1].stop - 1
                base_crop = base_mask[r_min:r_max+1, c_min:c_max+1]
                
                dilated = binary_dilation(base_mask, structure=np.ones((3,3)))
                adjacent_mask = dilated & ~base_mask & (inp > 0)
                
                adj_colors = np.unique(inp[adjacent_mask])
                
                for ac in adj_colors:
                    ac_mask = (inp == ac) & adjacent_mask
                    r_ac, c_ac = np.where(ac_mask)
                    if len(r_ac) == 1:
                        out_ac_mask = (out == ac)
                        slices_out = find_objects(out_ac_mask.astype(int))
                        if not slices_out: continue
                        s_out = slices_out[0]
                        r_min_out, r_max_out = s_out[0].start, s_out[0].stop - 1
                        c_min_out, c_max_out = s_out[1].start, s_out[1].stop - 1
                        out_ac_crop = out_ac_mask[r_min_out:r_max_out+1, c_min_out:c_max_out+1]
                        
                        matched = []
                        for k in range(8):
                            trans = transform(base_crop, k)
                            if trans.shape == out_ac_crop.shape and np.array_equal(trans, out_ac_crop):
                                matched.append(k)
                        
                        r_center = (r_min + r_max) / 2
                        c_center = (c_min + c_max) / 2
                        dr_c = r_ac[0] - r_center
                        dc_c = c_ac[0] - c_center
                        
                        print(f"Base {c}, attached {ac} at ({r_ac[0]}, {c_ac[0]}). Center ({r_center}, {c_center}).")
                        print(f"  dr_c={dr_c}, dc_c={dc_c}")
                        print(f"  Matched transforms: {matched}")

