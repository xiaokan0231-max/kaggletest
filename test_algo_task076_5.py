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
    
    m1 = x[:, 1:2]
    m2 = x[:, 2:3]
    m3 = x[:, 3:4]
    m4 = x[:, 4:5]
    
    m13 = torch.clamp(m1 + m3, 0, 1)
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
    
    final_out1 = torch.zeros_like(m1)
    final_out3 = torch.zeros_like(m3)
    
    S2 = m2
    m4_inv = 1.0 - m4
    
    for k_t in range(8):
        T4_k = transform(T4, k_t)
        T2_k = transform(T2, k_t)
        T1_k = transform(T1, k_t)
        T3_k = transform(T3, k_t)
        
        corr = F.conv2d(S2, T2_k, padding=(H-1, W-1))
        mismatch = F.conv2d(m4_inv, T4_k, padding=(H-1, W-1))
        
        valid_corr = corr * (mismatch < 0.5).float()
        
        kernel = torch.flip(valid_corr, [2, 3])
        
        shifted_T1 = F.conv2d(F.pad(T1_k, (W-1, W-1, H-1, H-1)), kernel)
        shifted_T3 = F.conv2d(F.pad(T3_k, (W-1, W-1, H-1, H-1)), kernel)
        
        final_out1 = torch.clamp(final_out1 + shifted_T1, 0, 1)
        final_out3 = torch.clamp(final_out3 + shifted_T3, 0, 1)
        
    out = x.clone()
    
    # Overwrite 0s
    mask_0 = x[:, 0:1]
    
    out1_add = final_out1 * mask_0
    out3_add = final_out3 * mask_0
    
    out[:, 1:2] = torch.clamp(out[:, 1:2] + out1_add, 0, 1)
    out[:, 3:4] = torch.clamp(out[:, 3:4] + out3_add, 0, 1)
    
    out[:, 0:1] = torch.clamp(out[:, 0:1] - out1_add - out3_add, 0, 1)
    
    return out

with open('neurogolf/data/raw/task076.json') as f:
    data = json.load(f)

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    # convert to one-hot
    H, W = inp.shape
    inp_t = torch.zeros(1, 10, H, W)
    for c in range(10):
        inp_t[0, c] = torch.tensor(inp == c, dtype=torch.float32)
        
    out_pred = solve(inp_t)
    out_pred_label = torch.argmax(out_pred, dim=1)[0].numpy()
    
    match = np.array_equal(out_pred_label, out)
    print(f"Train {i}: {match}")
    if not match:
        print("Expected:", out)
        print("Got:", out_pred_label)

