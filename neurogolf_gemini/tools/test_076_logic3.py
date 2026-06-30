import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task076.json") as f:
    task_data = json.load(f)

s = ndimage.generate_binary_structure(2, 2)

for i, ex in enumerate(task_data['train']):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    c_anchor = max(non_bg, key=lambda c: (inp == c).sum())
    
    anchor_mask = (inp == c_anchor)
    labeled_anchor, n_comp = ndimage.label(anchor_mask, structure=s)
    
    # Label all non-bg colors
    non_bg_mask = (inp != bg)
    labeled_non_bg, _ = ndimage.label(non_bg_mask, structure=s)
    
    max_touches = -1
    template_comp = -1
    
    for comp in range(1, n_comp + 1):
        mask = (labeled_anchor == comp)
        # Find which non_bg component this anchor belongs to
        non_bg_comp = labeled_non_bg[mask][0]
        
        full_comp_mask = (labeled_non_bg == non_bg_comp)
        
        touches = set(inp[full_comp_mask]) - {c_anchor, bg}
        if len(touches) > max_touches:
            max_touches = len(touches)
            template_comp = comp
            
    temp_anchor_mask = (labeled_anchor == template_comp)
    non_bg_comp = labeled_non_bg[temp_anchor_mask][0]
    full_comp_mask = (labeled_non_bg == non_bg_comp)
    
    temp_colors_mask = np.zeros_like(inp)
    for c in non_bg:
        if c != c_anchor:
            temp_colors_mask[full_comp_mask & (inp == c)] = c
            
    r, c = np.where(full_comp_mask)
    r_min, r_max = r.min(), r.max()
    c_min, c_max = c.min(), c.max()
    
    K_anchor = temp_anchor_mask[r_min:r_max+1, c_min:c_max+1]
    K_colors = temp_colors_mask[r_min:r_max+1, c_min:c_max+1]
    
    out_pred = inp.copy()
    
    input_hints = inp.copy()
    input_hints[inp == c_anchor] = bg
    input_hints[inp == bg] = 0
    
    A_sum = K_anchor.sum()
    
    def get_sym(A, C):
        syms = []
        for k in range(4):
            Ar = np.rot90(A, k)
            Cr = np.rot90(C, k)
            syms.append((Ar, Cr))
            syms.append((np.fliplr(Ar), np.fliplr(Cr)))
        return syms
        
    syms = get_sym(K_anchor, K_colors)
    
    for Ar, Cr in syms:
        kH, kW = Ar.shape
        for y in range(H - kH + 1):
            for x in range(W - kW + 1):
                if np.sum(anchor_mask[y:y+kH, x:x+kW] & Ar) == A_sum and np.sum(Ar) == A_sum:
                    hint_window = input_hints[y:y+kH, x:x+kW]
                    hint_mask = (hint_window != 0)
                    
                    hint_match = np.sum((hint_window == Cr) & hint_mask)
                    hint_total = np.sum(hint_mask)
                    
                    if hint_match == hint_total and hint_total > 0:
                        draw_mask = (Cr != 0)
                        out_pred[y:y+kH, x:x+kW][draw_mask] = Cr[draw_mask]
                        
    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
