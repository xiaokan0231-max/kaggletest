import torch
import torch.nn.functional as F
import json
import numpy as np

def transform(x, k):
    if k >= 4:
        x = torch.flip(x, [3])
        k = k - 4
    if k == 1:
        x = torch.rot90(x, 1, [2, 3])
    elif k == 2:
        x = torch.rot90(x, 2, [2, 3])
    elif k == 3:
        x = torch.rot90(x, 3, [2, 3])
    return x

def solve(x):
    B, _, H, W = x.shape
    
    counts = torch.sum(x, dim=(2,3))[0]
    bg_c = torch.argmax(counts).item()
    
    minority_cs = [c for c in range(10) if counts[c] > 0 and c != bg_c]
    
    k_dil_8 = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=x.device)
    
    c_path = -1
    for c in minority_cs:
        mask = x[:, c:c+1]
        flat_mask = mask.view(-1)
        idx = torch.argmax(flat_mask)
        seed = torch.zeros_like(flat_mask)
        seed[idx] = 1.0
        seed = seed.view(1, 1, H, W) * mask
        for _ in range(30):
            dilated = torch.clamp(F.conv2d(seed, k_dil_8, padding=1), 0, 1)
            seed = torch.clamp(dilated * mask, 0, 1)
        if torch.sum(seed) == torch.sum(mask):
            c_path = c
            break
            
    m_path = x[:, c_path:c_path+1]
    
    dilated_path = torch.clamp(F.conv2d(m_path, k_dil_8, padding=1), 0, 1)
    
    touches = []
    for c in minority_cs:
        if c != c_path:
            m_c = x[:, c:c+1]
            if torch.sum(dilated_path * m_c) > 0:
                touches.append(c)
                
    cA = touches[0]
    cB = touches[1]
    
    m_A = x[:, cA:cA+1]
    m_B = x[:, cB:cB+1]
    
    T_A_start = torch.clamp(dilated_path * m_A, 0, 1)
    T_B_start = torch.clamp(dilated_path * m_B, 0, 1)
    
    T_A = T_A_start
    T_B = T_B_start
    for _ in range(10):
        dilated_A = torch.clamp(F.conv2d(T_A, k_dil_8, padding=1), 0, 1)
        T_A = torch.clamp(dilated_A * m_A, 0, 1)
        dilated_B = torch.clamp(F.conv2d(T_B, k_dil_8, padding=1), 0, 1)
        T_B = torch.clamp(dilated_B * m_B, 0, 1)
        
    T_path = m_path
    
    final_path = m_path.clone()
    
    for s in range(1, 6):
        T_A_s = F.interpolate(T_A, scale_factor=s, mode='nearest')
        T_B_s = F.interpolate(T_B, scale_factor=s, mode='nearest')
        T_path_s = F.interpolate(T_path, scale_factor=s, mode='nearest')
        
        sum_A = T_A_s.sum()
        sum_B = T_B_s.sum()
        
        if sum_A == 0 or sum_B == 0:
            continue
            
        pad_h = s * H
        pad_w = s * W
        
        m_A_padded = F.pad(m_A, (pad_w, pad_w, pad_h, pad_h))
        m_B_padded = F.pad(m_B, (pad_w, pad_w, pad_h, pad_h))
            
        for k_t in range(8):
            T_A_k = transform(T_A_s, k_t)
            T_B_k = transform(T_B_s, k_t)
            T_path_k = transform(T_path_s, k_t)
            
            corr_A = F.conv2d(m_A_padded, T_A_k)
            corr_B = F.conv2d(m_B_padded, T_B_k)
            
            valid_A = (corr_A >= sum_A - 0.5)
            valid_B = (corr_B >= sum_B - 0.5)
            valid_shift = valid_A & valid_B
            
            kernel = torch.flip(valid_shift.float(), [2, 3])
            T_path_k_padded = F.pad(T_path_k, (W, W, H, H))
            shifted_path = F.conv2d(T_path_k_padded, kernel)
            
            final_path = torch.clamp(final_path + shifted_path, 0, 1)
            
    out = x.clone()
    
    mask_bg = x[:, bg_c:bg_c+1]
    path_add = final_path * mask_bg
    
    out[:, c_path:c_path+1] = torch.clamp(out[:, c_path:c_path+1] + path_add, 0, 1)
    out[:, bg_c:bg_c+1] = torch.clamp(out[:, bg_c:bg_c+1] - path_add, 0, 1)
    
    return out

with open('neurogolf/data/raw/task158.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    H, W = inp.shape
    inp_t = torch.zeros(1, 10, H, W)
    for c in range(10):
        inp_t[0, c] = torch.tensor(inp == c, dtype=torch.float32)
        
    out_pred = solve(inp_t)
    out_pred_label = torch.argmax(out_pred, dim=1)[0].numpy()
    
    match = np.array_equal(out_pred_label, out)
    print(f"Train {i}: {match}")
    if not match:
        diff = (out_pred_label != out)
        print("Expected:")
        print(out[diff])
        print("Got:")
        print(out_pred_label[diff])

