import torch
import torch.nn as nn
import onnx

class Task030Net(nn.Module):
    def forward(self, x):
        # x is (1, 10, 30, 30) one-hot
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        out = torch.zeros_like(x)
        
        # Anchor is color 1
        x_1 = x[:, 1:2, :, :]
        
        # Find the top row of anchor
        # V_1: (1, 1, 30, 1)
        V_1 = torch.max(x_1, dim=3, keepdim=True)[0]
        
        # We need the first row index where V_1 is 1
        # argmax returns the first index of the maximum value
        # If V_1 is all 0, r_1 will be 0.
        r_1 = torch.argmax(V_1.float(), dim=2, keepdim=True) # shape: (1, 1, 1, 1)
        
        # For each color C, we shift it to r_1
        for c in range(1, 10):
            x_c = x[:, c:c+1, :, :]
            V_c = torch.max(x_c, dim=3, keepdim=True)[0]
            r_c = torch.argmax(V_c.float(), dim=2, keepdim=True) # (1, 1, 1, 1)
            
            # T matrix: T[i, j] = 1 if r_1 - i == r_c - j
            i = torch.arange(30, dtype=torch.float32).view(1, 1, 30, 1)
            j = torch.arange(30, dtype=torch.float32).view(1, 1, 1, 30)
            
            # To make it compatible with ONNX tracing:
            r_1_float = r_1.float()
            r_c_float = r_c.float()
            
            # T shape: (1, 1, 30, 30)
            T = (r_1_float - i == r_c_float - j).float()
            
            # O_c = T @ x_c
            # x_c is (1, 1, 30, 30)
            # T is (1, 1, 30, 30)
            # matrix multiply the last two dimensions
            O_c = torch.matmul(T, x_c)
            
            # If x_c was empty, V_c is all 0, r_c is 0.
            # O_c will be shifted, but since x_c is all 0, O_c will be all 0.
            # So this is safe.
            out[:, c:c+1, :, :] = O_c
            
        # Background is where all other colors are 0
        max_color = torch.max(out[:, 1:, :, :], dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task030Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task030.onnx"
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
