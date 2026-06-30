import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task076.json") as f:
    task_data = json.load(f)

s = ndimage.generate_binary_structure(2, 2)
ex = task_data['train'][0]
inp = np.array(ex['input'])

colors, counts = np.unique(inp, return_counts=True)
bg = colors[np.argmax(counts)]
non_bg = [c for c in colors if c != bg]

c_anchor = 4
anchor_mask = (inp == c_anchor)
labeled, n_comp = ndimage.label(anchor_mask, structure=s)

template_comp = 1 # Top left anchor
temp_anchor_mask = (labeled == template_comp)
dilated_temp = ndimage.binary_dilation(temp_anchor_mask, structure=s)

temp_colors_mask = np.zeros_like(inp)
for c in non_bg:
    if c != c_anchor:
        temp_colors_mask[dilated_temp & (inp == c)] = c

r, c = np.where(dilated_temp)
r_min, r_max = r.min(), r.max()
c_min, c_max = c.min(), c.max()

K_anchor = temp_anchor_mask[r_min:r_max+1, c_min:c_max+1]
K_colors = temp_colors_mask[r_min:r_max+1, c_min:c_max+1]

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

input_hints = inp.copy()
input_hints[inp == c_anchor] = bg
input_hints[inp == bg] = 0

out_pred = inp.copy()

comp = 3
mask = (labeled == comp)
best_match_score = -1
best_draw = None

for i, (Ar, Cr) in enumerate(syms):
    kH, kW = Ar.shape
    for y in range(inp.shape[0] - kH + 1):
        for x in range(inp.shape[1] - kW + 1):
            if np.sum(anchor_mask[y:y+kH, x:x+kW] & Ar) == A_sum and np.sum(Ar) == A_sum:
                hint_window = input_hints[y:y+kH, x:x+kW]
                hint_mask = (hint_window != 0)
                
                hint_match = np.sum((hint_window == Cr) & hint_mask)
                hint_total = np.sum(hint_mask)
                
                if hint_match == hint_total and hint_total > 0:
                    print(f"Match found! Sym {i} at {y}, {x}. Score: {hint_total}")
                    best_match_score = hint_total
                    best_draw = (y, x, Cr)

if best_draw:
    y, x, Cr = best_draw
    print(f"Drawing at {y}, {x}")
    kH, kW = Cr.shape
    draw_mask = (Cr != 0)
    out_pred[y:y+kH, x:x+kW][draw_mask] = Cr[draw_mask]
    
    print("Pred window:")
    print(out_pred[y:y+kH, x:x+kW])
