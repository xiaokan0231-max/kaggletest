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
    
    # identify background
    counts = np.bincount(inp.flatten())
    bg = np.argmax(counts)
    
    for c in range(10):
        if c == bg: continue
        mask = (inp == c)
        if mask.sum() > 1:
            labeled, num = label(mask)
            for j in range(1, num+1):
                base_mask = (labeled == j)
                slices = find_objects(base_mask.astype(int))[0]
                rm, rM = slices[0].start, slices[0].stop
                cm, cM = slices[1].start, slices[1].stop
                base_crop = base_mask[rm:rM, cm:cM]
                
                dilated = binary_dilation(base_mask, structure=np.ones((3,3)))
                adjacent_mask = dilated & ~base_mask & (inp != bg) & (inp > 0)
                
                adj_colors = np.unique(inp[adjacent_mask])
                
                for ac in adj_colors:
                    ac_mask = (inp == ac) & adjacent_mask
                    r_ac, c_ac = np.where(ac_mask)
                    if len(r_ac) == 1:
                        out_ac_mask = (out == ac)
                        labeled_out, num_out = label(out_ac_mask)
                        
                        # Find which component in output is closest
                        best_k = -1
                        best_dr = 0
                        best_dc = 0
                        
                        for out_j in range(1, num_out+1):
                            out_comp = (labeled_out == out_j)
                            s_out = find_objects(out_comp.astype(int))[0]
                            rmo, rMo = s_out[0].start, s_out[0].stop
                            cmo, cMo = s_out[1].start, s_out[1].stop
                            out_crop = out_comp[rmo:rMo, cmo:cMo]
                            
                            for k in range(8):
                                trans = transform(base_crop, k)
                                if trans.shape == out_crop.shape and np.array_equal(trans, out_crop):
                                    best_k = k
                                    best_dr = rmo - rm
                                    best_dc = cmo - cm
                                    break
                                    
                        # pixel offset relative to center of bounding box
                        rc = (rm + rM - 1) / 2
                        cc = (cm + cM - 1) / 2
                        dpr = r_ac[0] - rc
                        dpc = c_ac[0] - cc
                        
                        # pixel offset relative to closest base pixel
                        r_base, c_base = np.where(base_mask)
                        dists = np.abs(r_base - r_ac[0]) + np.abs(c_base - c_ac[0])
                        idx = np.argmin(dists)
                        closest_r, closest_c = r_base[idx], c_base[idx]
                        
                        # Direction
                        dir_r = np.sign(r_ac[0] - closest_r)
                        dir_c = np.sign(c_ac[0] - closest_c)
                        
                        print(f"Base {c}->{ac}: Dir=({dir_r}, {dir_c}), dpr={dpr}, dpc={dpc} => k={best_k}, out_shift=({best_dr}, {best_dc})")

