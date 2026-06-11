import torch
import torch.nn as nn
import torch.nn.functional as F

class ARCModel(nn.Module):
    def forward(self, input):
        # input: (1, 10, 30, 30)
        
        color_indices = torch.arange(10, dtype=torch.float32, device=input.device).view(1, 10, 1, 1)
        padded = torch.sum(input * color_indices, dim=1).squeeze(0)  # (30, 30)
        
        valid_mask = torch.sum(input, dim=1).squeeze(0)  # (30, 30)
        
        valid_dy = torch.zeros(30, dtype=torch.bool, device=input.device)
        valid_dx = torch.zeros(30, dtype=torch.bool, device=input.device)
        
        for dy in range(1, 30):
            img1 = padded[:-dy, :]
            img2 = padded[dy:, :]
            m_valid = valid_mask[:-dy, :] * valid_mask[dy:, :]
            m_not0 = (img1 != 0) & (img2 != 0)
            mask = m_valid.bool() & m_not0
            conflict = (img1 != img2) & mask
            valid_dy[dy] = (conflict.sum() == 0)
            
        for dx in range(1, 30):
            img1 = padded[:, :-dx]
            img2 = padded[:, dx:]
            m_valid = valid_mask[:, :-dx] * valid_mask[:, dx:]
            m_not0 = (img1 != 0) & (img2 != 0)
            mask = m_valid.bool() & m_not0
            conflict = (img1 != img2) & mask
            valid_dx[dx] = (conflict.sum() == 0)
            
        best_dy = torch.argmax(valid_dy.int())
        best_dx = torch.argmax(valid_dx.int())
        
        best_dy = torch.clamp(best_dy, min=1)
        best_dx = torch.clamp(best_dx, min=1)
        
        Y, X = torch.meshgrid(torch.arange(30, device=input.device), torch.arange(30, device=input.device), indexing='ij')
        diff_y = torch.abs(Y - X)
        diff_x = torch.abs(Y - X)
        
        T_y = (diff_y % best_dy == 0).float()
        T_x = (diff_x % best_dx == 0).float()
        
        P = input[0].permute(1, 2, 0) # (30, 30, 10)
        scores = torch.einsum('ya,xb,abc->yxc', T_y, T_x, P)
        
        color_0_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=input.device)
        scores_no0 = scores * color_0_mask.view(1, 1, 10)
        
        best_color = torch.argmax(scores_no0, dim=-1)
        
        out_onehot = F.one_hot(best_color, 10).float()
        out = out_onehot.permute(2, 0, 1).unsqueeze(0)
        
        out = out * valid_mask.unsqueeze(0).unsqueeze(0)
        
        return out

if __name__ == '__main__':
    m = ARCModel()
    x = torch.zeros(1, 10, 30, 30)
    x[0, 1, 0:21, 0:21] = 1.0
    torch.onnx.export(
        m, 
        (x,), 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task017.onnx", 
        opset_version=14, 
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print("Model exported")
