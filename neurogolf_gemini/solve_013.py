import torch
import torch.nn as nn
import onnx

class Task013Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        H_tensor = torch.sum(torch.max(valid_mask, dim=3)[0], dim=2) # (1, 1)
        W_tensor = torch.sum(torch.max(valid_mask, dim=2)[0], dim=2) # (1, 1)
        is_wide = (W_tensor > H_tensor).float().view(1, 1, 1, 1)
        
        # --- Wide Branch ---
        V_wide = torch.max(x[:, 1:, :, :], dim=2)[0] # (1, 9, 30)
        V_wide_any = torch.max(V_wide, dim=1)[0] # (1, 30)
        c1_wide = torch.argmax(V_wide_any, dim=1, keepdim=True) # (1, 1)
        
        idx_30 = torch.arange(30, dtype=torch.float32, device=x.device).unsqueeze(0) # (1, 30)
        V_wide_masked = V_wide_any * (idx_30 != c1_wide.float()).float()
        c2_wide = torch.argmax(V_wide_masked, dim=1, keepdim=True) # (1, 1)
        
        D_wide = torch.abs(c1_wide - c2_wide).float()
        mod_wide = 2 * D_wide
        mod_wide_int = torch.clamp(mod_wide.to(torch.int64), min=1).view(1, 1, 1, 1)
        
        idx_c_int = torch.arange(30, dtype=torch.int64, device=x.device).view(1, 1, 1, 30)
        
        c1_mod = (c1_wide.view(1, 1, 1, 1).to(torch.int64) % mod_wide_int)
        c2_mod = (c2_wide.view(1, 1, 1, 1).to(torch.int64) % mod_wide_int)
        
        valid_c = (idx_c_int >= c1_wide.view(1, 1, 1, 1).to(torch.int64)).float()
        
        mask1_wide = ((idx_c_int % mod_wide_int) == c1_mod).float() * valid_c
        mask2_wide = ((idx_c_int % mod_wide_int) == c2_mod).float() * valid_c
        
        V_wide_all = torch.max(x, dim=2)[0] # (1, 10, 30)
        c1_wide_exp = c1_wide.view(1, 1, 1).expand(1, 10, 1)
        val1_wide = torch.gather(V_wide_all, 2, c1_wide_exp).unsqueeze(2) # (1, 10, 1, 1)
        
        c2_wide_exp = c2_wide.view(1, 1, 1).expand(1, 10, 1)
        val2_wide = torch.gather(V_wide_all, 2, c2_wide_exp).unsqueeze(2) # (1, 10, 1, 1)
        
        out_wide = val1_wide * mask1_wide + val2_wide * mask2_wide
        out_wide = out_wide.expand(1, 10, 30, 30)
        
        # --- Tall Branch ---
        V_tall = torch.max(x[:, 1:, :, :], dim=3)[0] # (1, 9, 30)
        V_tall_any = torch.max(V_tall, dim=1)[0] # (1, 30)
        r1_tall = torch.argmax(V_tall_any, dim=1, keepdim=True) # (1, 1)
        
        V_tall_masked = V_tall_any * (idx_30 != r1_tall.float()).float()
        r2_tall = torch.argmax(V_tall_masked, dim=1, keepdim=True) # (1, 1)
        
        D_tall = torch.abs(r1_tall - r2_tall).float()
        mod_tall = 2 * D_tall
        mod_tall_int = torch.clamp(mod_tall.to(torch.int64), min=1).view(1, 1, 1, 1)
        
        idx_r_int = torch.arange(30, dtype=torch.int64, device=x.device).view(1, 1, 30, 1)
        
        r1_mod = (r1_tall.view(1, 1, 1, 1).to(torch.int64) % mod_tall_int)
        r2_mod = (r2_tall.view(1, 1, 1, 1).to(torch.int64) % mod_tall_int)
        
        valid_r = (idx_r_int >= r1_tall.view(1, 1, 1, 1).to(torch.int64)).float()
        
        mask1_tall = ((idx_r_int % mod_tall_int) == r1_mod).float() * valid_r
        mask2_tall = ((idx_r_int % mod_tall_int) == r2_mod).float() * valid_r
        
        V_tall_all = torch.max(x, dim=3)[0] # (1, 10, 30)
        r1_tall_exp = r1_tall.view(1, 1, 1).expand(1, 10, 1)
        val1_tall = torch.gather(V_tall_all, 2, r1_tall_exp).unsqueeze(3) # (1, 10, 1, 1)
        
        r2_tall_exp = r2_tall.view(1, 1, 1).expand(1, 10, 1)
        val2_tall = torch.gather(V_tall_all, 2, r2_tall_exp).unsqueeze(3) # (1, 10, 1, 1)
        
        out_tall = val1_tall * mask1_tall + val2_tall * mask2_tall
        out_tall = out_tall.expand(1, 10, 30, 30)
        
        # --- Combine ---
        out = out_wide * is_wide + out_tall * (1.0 - is_wide)
        
        # Background
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1.0 - torch.clamp(max_color, 0.0, 1.0)
        out[:, 0:1, :, :] = bg
        
        return out * valid_mask

def export_onnx():
    net = Task013Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task013.onnx"
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
