import torch
import torch.nn as nn
import onnx

class Task008Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = torch.zeros_like(x)
        
        # Color 8 is fixed
        x_8 = x[:, 8:9, :, :]
        out[:, 8:9, :, :] = x_8
        
        # Color 2 moves to touch 8
        x_2 = x[:, 2:3, :, :]
        
        # We need to find the shift for x_2.
        # Instead of complex logic, let's just evaluate ALL possible shifts (dr, dc)
        # For a shift (dr, dc), x_2_shifted touches x_8 if:
        # x_2_shifted dilated by 1 intersects x_8.
        # And we want the shift that is in the "direction" of 8.
        # But actually, 2 moves either horizontally or vertically.
        # We can just simulate the move!
        # A simple way to simulate gravity:
        # Iteratively move x_2 towards x_8 by 1 pixel until it touches.
        # In ONNX, we can unroll the loop 30 times!
        
        curr_2 = x_2
        
        # Unroll 30 times
        for _ in range(30):
            # Find the center of curr_2 and x_8
            # We can use bounding box min/max
            
            # Row and Col indices
            r_idx = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1)
            c_idx = torch.arange(30, dtype=torch.float32).view(1, 1, 1, 30)
            
            # curr_2 bounding box
            mask_2 = (curr_2 > 0).float()
            r_2 = torch.sum(r_idx * mask_2) / (torch.sum(mask_2) + 1e-5)
            c_2 = torch.sum(c_idx * mask_2) / (torch.sum(mask_2) + 1e-5)
            
            # x_8 bounding box
            mask_8 = (x_8 > 0).float()
            r_8 = torch.sum(r_idx * mask_8) / (torch.sum(mask_8) + 1e-5)
            c_8 = torch.sum(c_idx * mask_8) / (torch.sum(mask_8) + 1e-5)
            
            # Determine direction
            dr = torch.sign(r_8 - r_2)
            dc = torch.sign(c_8 - c_2)
            
            # If dr is larger in magnitude, move vertically. Else horizontally.
            # Wait, in the examples, 2 is either purely horizontal or vertical.
            # So one of dr, dc is non-zero, the other is 0.
            # Let's just allow diagonal if necessary, but we only move if it doesn't overlap 8.
            
            # To move curr_2 by (dr, dc), we can use a translation matrix.
            # Actually, we can use 2D convolution for a 1-pixel shift!
            # But Conv2d weights must be static.
            # So we can construct a translation matrix.
            # Wait! dr and dc are dynamic! We can't easily construct a matrix with dynamic dr, dc.
            
            # Alternative: Since there's only one x_2 and one x_8, we can just find the distance.
            pass
            
        # Let's do a static approach:
        # Compute the bounding boxes of 2 and 8
        r_idx = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1)
        c_idx = torch.arange(30, dtype=torch.float32).view(1, 1, 1, 30)
        
        # min and max row/col for 2
        # If no 2, set min to 30, max to -1
        mask_2 = (x_2 > 0).float()
        r_2_min = torch.min(torch.where(mask_2 > 0, r_idx, torch.tensor(30.0)))
        r_2_max = torch.max(torch.where(mask_2 > 0, r_idx, torch.tensor(-1.0)))
        c_2_min = torch.min(torch.where(mask_2 > 0, c_idx, torch.tensor(30.0)))
        c_2_max = torch.max(torch.where(mask_2 > 0, c_idx, torch.tensor(-1.0)))
        
        mask_8 = (x_8 > 0).float()
        r_8_min = torch.min(torch.where(mask_8 > 0, r_idx, torch.tensor(30.0)))
        r_8_max = torch.max(torch.where(mask_8 > 0, r_idx, torch.tensor(-1.0)))
        c_8_min = torch.min(torch.where(mask_8 > 0, c_idx, torch.tensor(30.0)))
        c_8_max = torch.max(torch.where(mask_8 > 0, c_idx, torch.tensor(-1.0)))
        
        # Calculate shifts
        # If 2 is above 8, it moves down to touch 8. Shift dr = r_8_min - r_2_max - 1
        # If 2 is below 8, it moves up. Shift dr = r_8_max - r_2_min + 1
        # If 2 is left of 8, it moves right. Shift dc = c_8_min - c_2_max - 1
        # If 2 is right of 8, it moves left. Shift dc = c_8_max - c_2_min + 1
        
        is_above = (r_2_max < r_8_min).float()
        is_below = (r_2_min > r_8_max).float()
        is_left  = (c_2_max < c_8_min).float()
        is_right = (c_2_min > c_8_max).float()
        
        # Wait, if it's strictly above and column overlaps, it's above.
        # Let's check overlap
        col_overlap = ((c_2_max >= c_8_min) & (c_2_min <= c_8_max)).float()
        row_overlap = ((r_2_max >= r_8_min) & (r_2_min <= r_8_max)).float()
        
        # Apply shift only if they overlap in the other dimension
        dr = is_above * col_overlap * (r_8_min - r_2_max - 1) + \
             is_below * col_overlap * (r_8_max - r_2_min + 1)
             
        dc = is_left * row_overlap * (c_8_min - c_2_max - 1) + \
             is_right * row_overlap * (c_8_max - c_2_min + 1)
             
        # Create Translation matrix T_r and T_c
        i = torch.arange(30, dtype=torch.float32).view(30, 1)
        j = torch.arange(30, dtype=torch.float32).view(1, 30)
        
        T_r = (i - j == dr).float().unsqueeze(0).unsqueeze(0) # (1, 1, 30, 30)
        T_c = (i - j == dc).float().unsqueeze(0).unsqueeze(0) # (1, 1, 30, 30)
        
        # Apply T_r: O = T_r @ x_2
        O_2 = torch.matmul(T_r, x_2)
        # Apply T_c: O = O @ T_c.T  (since it acts on columns)
        O_2 = torch.matmul(O_2, T_c.transpose(2, 3))
        
        out[:, 2:3, :, :] = O_2
        
        # Background
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task008Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task008.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
