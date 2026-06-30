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
    B, C, H, W = x.shape
    
    # identify bg:
    counts = torch.sum(x, dim=(2,3))[0]
    bg_c = torch.argmax(counts).item()
    
    # identify minority colors:
    minority_cs = [c for c in range(10) if counts[c] > 0 and c != bg_c]
    
    # We need to find the connecting shape.
    # It touches the other two colors.
    # Let's do dilation on each minority color to see which touches BOTH others.
    k_dil = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32, device=x.device)
    
    c_path = -1
    cA = -1
    cB = -1
    
    for c in minority_cs:
        mask_c = x[:, c:c+1]
        dilated_c = torch.clamp(F.conv2d(mask_c, k_dil, padding=1), 0, 1)
        touches = []
        for other_c in minority_cs:
            if other_c != c:
                mask_other = x[:, other_c:other_c+1]
                if torch.sum(dilated_c * mask_other) > 0:
                    touches.append(other_c)
        if len(touches) == 2:
            c_path = c
            cA = touches[0]
            cB = touches[1]
            break
            
    m_path = x[:, c_path:c_path+1]
    m_A = x[:, cA:cA+1]
    m_B = x[:, cB:cB+1]
    
    # The template is the SPECIFIC component of m_A and m_B that touches m_path.
    dilated_path = torch.clamp(F.conv2d(m_path, k_dil, padding=1), 0, 1)
    T_A_start = torch.clamp(dilated_path * m_A, 0, 1)
    T_B_start = torch.clamp(dilated_path * m_B, 0, 1)
    
    # propagate to get full T_A and T_B
    T_A = T_A_start
    T_B = T_B_start
    for _ in range(10):
        dilated_A = torch.clamp(F.conv2d(T_A, k_dil, padding=1), 0, 1)
        T_A = torch.clamp(dilated_A * m_A, 0, 1)
        dilated_B = torch.clamp(F.conv2d(T_B, k_dil, padding=1), 0, 1)
        T_B = torch.clamp(dilated_B * m_B, 0, 1)
        
    T_path = m_path
    
    final_path = torch.zeros_like(m_path)
    
    for s in range(1, 6):
        # Scale template
        T_A_s = F.interpolate(T_A, scale_factor=s, mode='nearest')
        T_B_s = F.interpolate(T_B, scale_factor=s, mode='nearest')
        T_path_s = F.interpolate(T_path, scale_factor=s, mode='nearest')
        
        sum_A = T_A_s.sum()
        sum_B = T_B_s.sum()
        
        if sum_A == 0 or sum_B == 0:
            continue
            
        for k_t in range(8):
            T_A_k = transform(T_A_s, k_t)
            T_B_k = transform(T_B_s, k_t)
            T_path_k = transform(T_path_s, k_t)
            
            # Since F.interpolate scales H, W to s*H, s*W.
            # We want to use F.conv2d to find match.
            # But m_A and m_B are size H, W.
            # T_A_k is size sH, sW.
            # Wait! If we convolve m_A with T_A_k, m_A MUST BE PADDED!
            # The kernel is T_A_k.
            # m_A is H x W. Kernel is sH x sW.
            # If s > 1, Kernel is LARGER than m_A!
            # F.conv2d requires input >= kernel!
            pass

