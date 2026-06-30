import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task076.json") as f:
    task_data = json.load(f)

s = ndimage.generate_binary_structure(2, 2)

for i, ex in enumerate(task_data['train']):
    if i == 1: continue
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    c_anchor = max(non_bg, key=lambda c: (inp == c).sum())
    
    anchor_mask = (inp == c_anchor)
    labeled_anchor, n_comp = ndimage.label(anchor_mask, structure=s)
    
    non_bg_mask = (inp != bg)
    labeled_non_bg, _ = ndimage.label(non_bg_mask, structure=s)
    
    max_touches = -1
    template_comp = -1
    for comp in range(1, n_comp + 1):
        mask = (labeled_anchor == comp)
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
            
    T_anchor = temp_anchor_mask.astype(int)
    T_colors = temp_colors_mask
    
    A_sum = T_anchor.sum()
    
    def get_sym(A, C):
        syms = []
        for k in range(4):
            Ar = np.rot90(A, k)
            Cr = np.rot90(C, k)
            syms.append((Ar, Cr))
            syms.append((np.fliplr(Ar), np.fliplr(Cr)))
        return syms
        
    syms = get_sym(T_anchor, T_colors)
    
    input_hints = inp.copy()
    input_hints[inp == c_anchor] = 0
    input_hints[inp == bg] = 0
    
    out_pred = inp.copy()
    
    for Ar, Cr in syms:
        for dy in range(-H + 1, H):
            for dx in range(-W + 1, W):
                y1, y2 = max(0, dy), min(H, H+dy)
                x1, x2 = max(0, dx), min(W, W+dx)
                
                ty1, ty2 = max(0, -dy), min(H, H-dy)
                tx1, tx2 = max(0, -dx), min(W, W-dx)
                
                if y2 <= y1 or x2 <= x1:
                    continue
                    
                A_view = anchor_mask[y1:y2, x1:x2]
                Ar_view = Ar[ty1:ty2, tx1:tx2]
                
                if np.sum(A_view & Ar_view) == A_sum and np.sum(Ar_view) == A_sum:
                    H_view = input_hints[y1:y2, x1:x2]
                    Cr_view = Cr[ty1:ty2, tx1:tx2]
                    
                    hint_mask = (H_view != 0)
                    support_mask = (Cr_view != 0)
                    
                    hint_match = np.sum((H_view == Cr_view) & hint_mask & support_mask)
                    support_overlap = np.sum(hint_mask & support_mask)
                    
                    if hint_match > 0 and support_overlap == hint_match:
                        draw_mask = (Cr_view != 0)
                        out_pred[y1:y2, x1:x2][draw_mask] = Cr_view[draw_mask]
                        
    diff = np.where(out_pred != out_expected)
    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
    for dy, dx in zip(*diff):
        print(f"({dy}, {dx}): pred {out_pred[dy, dx]}, exp {out_expected[dy, dx]}")
