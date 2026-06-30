import json
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
import numpy as np

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
            
    r, c = np.where(full_comp_mask)
    r_min, r_max = r.min(), r.max()
    c_min, c_max = c.min(), c.max()
    
    K_anchor = temp_anchor_mask[r_min:r_max+1, c_min:c_max+1].astype(np.float32)
    K_colors = temp_colors_mask[r_min:r_max+1, c_min:c_max+1]
    
    K_colors_onehot = np.zeros((10, K_colors.shape[0], K_colors.shape[1]), dtype=np.float32)
    for c in range(10):
        K_colors_onehot[c] = (K_colors == c)
    K_colors_onehot[0] = 0
    
    A_sum = K_anchor.sum()
    
    def get_sym(A, C):
        syms = []
        for k in range(4):
            Ar = np.rot90(A, k).copy()
            Cr = np.rot90(C, k, axes=(1, 2)).copy()
            syms.append((Ar, Cr))
            syms.append((np.fliplr(Ar).copy(), np.flip(Cr, axis=2).copy()))
        return syms
        
    syms = get_sym(K_anchor, K_colors_onehot)
    
    inp_t = torch.tensor(inp, dtype=torch.long)
    inp_onehot = F.one_hot(inp_t, num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
    
    A_input = inp_onehot[:, c_anchor:c_anchor+1]
    H_input = inp_onehot.clone()
    H_input[:, c_anchor] = 0
    H_input[:, bg] = 0
    
    H_mask = (H_input.sum(dim=1, keepdim=True) > 0.5).float()
    
    out_pred = inp_onehot.clone()
    
    for Ar, Cr in syms:
        Ar_t = torch.tensor(Ar).unsqueeze(0).unsqueeze(0)
        Cr_t = torch.tensor(Cr).unsqueeze(0)
        
        kH, kW = Ar.shape
        
        match_A = F.conv2d(A_input, Ar_t)
        is_anchor_match = (match_A > A_sum - 0.5)
        
        hint_match = F.conv2d(H_input, Cr_t)
        ones_k = torch.ones(1, 1, kH, kW)
        hint_total = F.conv2d(H_mask, ones_k)
        
        is_hint_match = (torch.abs(hint_match - hint_total) < 0.5) & (hint_total > 0.5)
        
        valid_match = (is_anchor_match & is_hint_match).float()
        
        if valid_match.sum() > 0:
            drawn = F.conv_transpose2d(valid_match, Cr_t)
            out_pred = torch.max(out_pred, drawn)
            
    out_pred[:, bg] = torch.where(out_pred[:, 1:].max(dim=1)[0] > 0.5, 0.0, out_pred[:, bg])
    out_final = torch.argmax(out_pred, dim=1).squeeze(0).numpy()
    print(f"Ex {i} Match: {np.array_equal(out_final, out_expected)}")
