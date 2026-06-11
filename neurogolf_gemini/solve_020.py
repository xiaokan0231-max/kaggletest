import torch
import torch.nn as nn
import onnx

class Task020Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # Find any non-zero pixel
        x_any = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0] # (1, 1, H, W)
        
        r_any = torch.max(x_any, dim=3, keepdim=True)[0] # (1, 1, H, 1)
        c_any = torch.max(x_any, dim=2, keepdim=True)[0] # (1, 1, 1, W)
        
        idx_r = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1)
        idx_c = torch.arange(30, dtype=torch.float32).view(1, 1, 1, 30)
        
        min_r = torch.min(torch.where(r_any > 0, idx_r, torch.tensor(30.0)))
        max_r = torch.max(torch.where(r_any > 0, idx_r, torch.tensor(-1.0)))
        
        min_c = torch.min(torch.where(c_any > 0, idx_c, torch.tensor(30.0)))
        max_c = torch.max(torch.where(c_any > 0, idx_c, torch.tensor(-1.0)))
        
        Sr = min_r + max_r
        Sc = min_c + max_c
        Ssum = (Sr + Sc) / 2
        Sdiff = (Sc - Sr) / 2
        
        u = torch.arange(30, dtype=torch.float32).view(30, 1)
        v = torch.arange(30, dtype=torch.float32).view(1, 30)
        
        Mr_180 = (v == Sr - u).float().unsqueeze(0).unsqueeze(0)
        Mc_180 = (u == Sc - v).float().unsqueeze(0).unsqueeze(0)
        
        Mr_90 = (v == u + Sdiff).float().unsqueeze(0).unsqueeze(0)
        Mc_90 = (u == Ssum - v).float().unsqueeze(0).unsqueeze(0)
        
        Mr_270 = (v == Ssum - u).float().unsqueeze(0).unsqueeze(0)
        Mc_270 = (u == v - Sdiff).float().unsqueeze(0).unsqueeze(0)
        
        out = torch.zeros_like(x)
        XT = x.transpose(2, 3)
        
        for c in range(1, 10):
            xc = x[:, c:c+1, :, :]
            xcT = XT[:, c:c+1, :, :]
            
            X_180 = torch.matmul(Mr_180, torch.matmul(xc, Mc_180))
            X_90 = torch.matmul(Mr_90, torch.matmul(xcT, Mc_90))
            X_270 = torch.matmul(Mr_270, torch.matmul(xcT, Mc_270))
            
            X_flip_r = torch.matmul(Mr_180, xc)
            X_flip_c = torch.matmul(xc, Mc_180)
            
            Oc = torch.max(xc, X_180)
            Oc = torch.max(Oc, X_90)
            Oc = torch.max(Oc, X_270)
            Oc = torch.max(Oc, X_flip_r)
            Oc = torch.max(Oc, X_flip_c)
            
            out[:, c:c+1, :, :] = Oc
            
        # Background
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task020Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task020.onnx"
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
