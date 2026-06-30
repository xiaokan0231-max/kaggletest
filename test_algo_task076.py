import torch
import torch.nn.functional as F

def transform(x, k):
    # k in 0..7
    # 0: identity
    # 1: rot90
    # 2: rot180
    # 3: rot270
    # 4: flip H
    # 5: flip H, rot90
    # 6: flip H, rot180
    # 7: flip H, rot270
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
    # x is [B, 10, H, W]
    B, _, H, W = x.shape
    
    m1 = x[:, 1:2]
    m2 = x[:, 2:3]
    m3 = x[:, 3:4]
    m4 = x[:, 4:5]
    
    # 1. Isolate template
    m13 = torch.clamp(m1 + m3, 0, 1)
    # dilate m13
    k = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32, device=x.device)
    dilated_m13 = torch.clamp(F.conv2d(m13, k, padding=1), 0, 1)
    
    adj_4 = torch.clamp(dilated_m13 * m4, 0, 1)
    
    T4 = adj_4
    for _ in range(30):
        dilated_T4 = torch.clamp(F.conv2d(T4, k, padding=1), 0, 1)
        T4 = torch.clamp(dilated_T4 * m4, 0, 1)
        
    dilated_T4 = torch.clamp(F.conv2d(T4, k, padding=1), 0, 1)
    T2 = torch.clamp(dilated_T4 * m2, 0, 1)
    T1 = m1
    T3 = m3
    
    out1 = m1.clone()
    out3 = m3.clone()
    
    # Coordinates
    r_grid = torch.arange(H, device=x.device).view(1, 1, H, 1).float()
    c_grid = torch.arange(W, device=x.device).view(1, 1, 1, W).float()
    
    # Targets
    S2 = m2  # All 2s
    
    for k_t in range(8):
        T4_k = transform(T4, k_t)
        T2_k = transform(T2, k_t)
        T1_k = transform(T1, k_t)
        T3_k = transform(T3, k_t)
        
        # Find (r_k, c_k) for T2_k
        # If T2_k is empty (shouldn't be, but just in case), sum is 0
        sum_T2 = torch.sum(T2_k, dim=(2,3), keepdim=True)
        # Avoid division by zero
        r_k = torch.sum(r_grid * T2_k, dim=(2,3), keepdim=True) / (sum_T2 + 1e-6)
        c_k = torch.sum(c_grid * T2_k, dim=(2,3), keepdim=True) / (sum_T2 + 1e-6)
        
        # For each pixel in grid, we consider it as a potential S2
        # dr = r_grid - r_k  (shape: B, 1, H, 1)
        # dc = c_grid - c_k  (shape: B, 1, 1, W)
        
        # We want to shift T4_k by (dr, dc) for EVERY pixel in the grid.
        # This means for a potential S2 at (r_i, c_i), the shift is (r_i - r_k, c_i - c_k).
        # The shifted T4_k value at (r, c) comes from T4_k at (r - dr, c - dc).
        # = T4_k(r - r_i + r_k, c - c_i + c_k)
        
        # Let's do this via F.grid_sample or by broadcasting.
        # Broadcasting is easier and fully ONNX compatible without grid_sample issues.
        # r_idx = r_grid.view(1, 1, H, 1, 1) - r_grid.view(1, 1, 1, H, 1) + r_k.view(B, 1, 1, 1, 1)
        # Wait, let's just use F.conv2d!
        # If we use S2 as an image, and T4_k as kernel?
        # Actually, if we just want to apply the shift and check:
        pass

