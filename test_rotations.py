import json
import numpy as np
from scipy.ndimage import label, find_objects

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
    
    counts = np.bincount(inp.flatten())
    bg = np.argmax(counts)
    
    # components
    mask = (inp != bg) & (inp > 0)
    labeled, num = label(mask, structure=np.ones((3,3)))
    
    print(f"\n--- Train {i} ---")
    for j in range(1, num+1):
        comp = (labeled == j)
        comp_colors = inp[comp]
        color_counts = np.bincount(comp_colors)
        base_color = np.argmax(color_counts)
        
        base_mask = comp & (inp == base_color)
        s = find_objects(base_mask.astype(int))[0]
        base_crop = base_mask[s[0].start:s[0].stop, s[1].start:s[1].stop]
        
        print(f"Component {j}: base color {base_color}")
        
        # for each pixel in component
        r, c = np.where(comp)
        for pr, pc in zip(r, c):
            color = inp[pr, pc]
            
            # find the block in output
            # the block should be exactly transform(base_crop, k)
            # just search for it
            out_mask = (out == color)
            labeled_out, num_out = label(out_mask)
            
            best_k = -1
            for oj in range(1, num_out+1):
                o_comp = (labeled_out == oj)
                os = find_objects(o_comp.astype(int))[0]
                o_crop = o_comp[os[0].start:os[0].stop, os[1].start:os[1].stop]
                
                for k in range(8):
                    tk = transform(base_crop, k)
                    if tk.shape == o_crop.shape and np.array_equal(tk, o_crop):
                        best_k = k
                        break
                if best_k != -1: break
                
            print(f"  Pixel {color} at ({pr}, {pc}) -> k={best_k}")

