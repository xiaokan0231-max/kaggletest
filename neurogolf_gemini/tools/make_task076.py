import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage as ndimage
import json
import os

class Task076(nn.Module):
    def __init__(self, c_anchor, bg_color, syms1, syms2):
        super().__init__()
        self.c_anchor = c_anchor
        self.bg_color = bg_color
        
        Ar_stack1 = torch.tensor(np.stack([a for a, c in syms1])).unsqueeze(1).float()
        Cr_stack1 = torch.tensor(np.stack([c for a, c in syms1])).float()
        self.register_buffer('Ar_stack1', Ar_stack1)
        self.register_buffer('Cr_stack1', Cr_stack1)
        
        self.has_syms2 = len(syms2) > 0
        if self.has_syms2:
            Ar_stack2 = torch.tensor(np.stack([a for a, c in syms2])).unsqueeze(1).float()
            Cr_stack2 = torch.tensor(np.stack([c for a, c in syms2])).float()
            self.register_buffer('Ar_stack2', Ar_stack2)
            self.register_buffer('Cr_stack2', Cr_stack2)
        
        self.A_sum = float(syms1[0][0].sum())

    def forward(self, x):
        _, _, orig_H, orig_W = x.shape
        x_pad = F.pad(x, (0, 30 - orig_W, 0, 30 - orig_H))
        
        A_input = x_pad[:, self.c_anchor:self.c_anchor+1]
        H_input = x_pad.clone()
        H_input[:, self.c_anchor] = 0
        H_input[:, self.bg_color] = 0
        
        H_mask = (H_input.sum(dim=1, keepdim=True) > 0.5).float()
        
        out_pred = x_pad.clone()
        
        # Stack 1
        match_A1 = F.conv2d(A_input, self.Ar_stack1)
        hint_match1 = F.conv2d(H_input, self.Cr_stack1)
        kH1, kW1 = self.Ar_stack1.shape[2], self.Ar_stack1.shape[3]
        hint_total1 = F.conv2d(H_mask, torch.ones(1, 1, kH1, kW1, device=x.device))
        
        is_anchor_match1 = (match_A1 > self.A_sum - 0.5)
        is_hint_match1 = (torch.abs(hint_match1 - hint_total1) < 0.5) & (hint_total1 > 0.5)
        valid_match1 = (is_anchor_match1 & is_hint_match1).float()
        drawn1 = F.conv_transpose2d(valid_match1, self.Cr_stack1)
        out_pred = torch.max(out_pred, drawn1)
        
        # Stack 2
        if self.has_syms2:
            match_A2 = F.conv2d(A_input, self.Ar_stack2)
            hint_match2 = F.conv2d(H_input, self.Cr_stack2)
            kH2, kW2 = self.Ar_stack2.shape[2], self.Ar_stack2.shape[3]
            hint_total2 = F.conv2d(H_mask, torch.ones(1, 1, kH2, kW2, device=x.device))
            
            is_anchor_match2 = (match_A2 > self.A_sum - 0.5)
            is_hint_match2 = (torch.abs(hint_match2 - hint_total2) < 0.5) & (hint_total2 > 0.5)
            valid_match2 = (is_anchor_match2 & is_hint_match2).float()
            drawn2 = F.conv_transpose2d(valid_match2, self.Cr_stack2)
            out_pred = torch.max(out_pred, drawn2)
            
        out_pred[:, self.bg_color] = torch.where(out_pred[:, 1:].max(dim=1)[0] > 0.5, 0.0, out_pred[:, self.bg_color])
        out_final = torch.argmax(out_pred, dim=1, keepdim=True).float()
        out_one_hot = (torch.arange(10, device=x.device).view(1, 10, 1, 1).float() == out_final).float()
        
        return out_one_hot[:, :, :orig_H, :orig_W]

if __name__ == "__main__":
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
    
    def get_sym(A, C):
        syms1 = []
        syms2 = []
        for k in range(4):
            Ar = np.rot90(A, k).copy()
            Cr = np.rot90(C, k, axes=(1, 2)).copy()
            if Ar.shape == A.shape:
                syms1.append((Ar, Cr))
                syms1.append((np.fliplr(Ar).copy(), np.flip(Cr, axis=2).copy()))
            else:
                syms2.append((Ar, Cr))
                syms2.append((np.fliplr(Ar).copy(), np.flip(Cr, axis=2).copy()))
        return syms1, syms2
        
    syms1, syms2 = get_sym(K_anchor, K_colors_onehot)
    
    model = Task076(int(c_anchor), int(bg), syms1, syms2)
    model.eval()
    
    inp_t = torch.tensor(inp, dtype=torch.long)
    inp_onehot = F.one_hot(inp_t, num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
    
    out_t = model(inp_onehot)
    print("Python Output:")
    print(torch.argmax(out_t, dim=1).squeeze(0).numpy())
    
    # Export ONNX
    dummy_input = torch.zeros((1, 10, 10, 14))
    dummy_input[0, 0, :, :] = 1.0
    
    torch.onnx.export(
        model, 
        dummy_input, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task076.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        opset_version=17
    )
    print("Exported task076.onnx")
