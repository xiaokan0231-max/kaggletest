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
c_anchor = max(non_bg, key=lambda c: (inp == c).sum())
anchor_mask = (inp == c_anchor)
labeled, n_comp = ndimage.label(anchor_mask, structure=s)

# template
max_touches = -1
template_comp = -1
for comp in range(1, n_comp + 1):
    mask = (labeled == comp)
    dilated = ndimage.binary_dilation(mask, structure=s)
    touches = set(inp[dilated]) - {c_anchor, bg}
    if len(touches) > max_touches:
        max_touches = len(touches)
        template_comp = comp

temp_anchor_mask = (labeled == template_comp)
dilated_temp = ndimage.binary_dilation(temp_anchor_mask, structure=s)

temp_colors_mask = np.zeros_like(inp)
for c in non_bg:
    if c != c_anchor:
        temp_colors_mask[dilated_temp & (inp == c)] = c

r, c = np.where(dilated_temp)
K_anchor = temp_anchor_mask[r.min():r.max()+1, c.min():c.max()+1]
K_colors = temp_colors_mask[r.min():r.max()+1, c.min():c.max()+1]

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

# target
comp = 3  # bottom left anchor
mask = (labeled == comp)
print("Target 3 anchor at:", np.where(mask))

for i, (Ar, Cr) in enumerate(syms):
    kH, kW = Ar.shape
    for y in range(inp.shape[0] - kH + 1):
        for x in range(inp.shape[1] - kW + 1):
            if np.sum(anchor_mask[y:y+kH, x:x+kW] & Ar) == A_sum and np.sum(Ar) == A_sum:
                # To only check the match at the correct target location:
                if np.sum(mask[y:y+kH, x:x+kW] & Ar) == A_sum:
                    hint_window = input_hints[y:y+kH, x:x+kW]
                    hint_mask = (hint_window != 0)
                    hint_match = np.sum((hint_window == Cr) & hint_mask)
                    hint_total = np.sum(hint_mask)
                    print(f"Sym {i} at {y}, {x}: hint_total={hint_total}, hint_match={hint_match}")
