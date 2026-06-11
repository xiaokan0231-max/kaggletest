import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task012Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.K_cross = nn.Parameter(torch.tensor([[[[0, 1, 0],
                                                    [1, 0, 1],
                                                    [0, 1, 0]]]], dtype=torch.float32), requires_grad=False)
        
        self.K_X = nn.Parameter(torch.tensor([[[[1, 0, 0, 0, 1],
                                                [0, 1, 0, 1, 0],
                                                [0, 0, 1, 0, 0],
                                                [0, 1, 0, 1, 0],
                                                [1, 0, 0, 0, 1]]]], dtype=torch.float32), requires_grad=False)
        
        self.K_plus = nn.Parameter(torch.tensor([[[[0, 0, 1, 0, 0],
                                                   [0, 0, 1, 0, 0],
                                                   [1, 1, 0, 1, 1],
                                                   [0, 0, 1, 0, 0],
                                                   [0, 0, 1, 0, 0]]]], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # any non-zero pixel
        x_any = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0] # (1, 1, H, W)
        
        # Find centers
        N = F.conv2d(x_any, self.K_cross, padding=1)
        M_center = (N == 4).float() * x_any
        
        out = torch.zeros_like(x)
        
        for c in range(1, 10):
            xc = x[:, c:c+1, :, :]
            
            C1_c = M_center * xc
            O1_c = (F.conv2d(C1_c, self.K_X, padding=2) > 0).float()
            
            C2_c = xc * (1 - M_center)
            Centers_for_C2 = (F.conv2d(C2_c, self.K_cross, padding=1) > 0).float() * M_center
            O2_c = (F.conv2d(Centers_for_C2, self.K_plus, padding=2) > 0).float()
            
            Oc = torch.max(O1_c, O2_c)
            out[:, c:c+1, :, :] = Oc
            
        # Background
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task012Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task012.onnx"
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
