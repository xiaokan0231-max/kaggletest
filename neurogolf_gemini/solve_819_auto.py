import torch
import torch.nn as nn
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        # map: array of size 10 where index is input color, value is output color
        mapping = [0]*10

        for i in range(10):
            if mapping[i] == 0 and i not in [k for k in mapping.keys()]:
                mapping[i] = i
                
        self.mapping = nn.Parameter(torch.tensor(mapping, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # x is (1, 10, H, W) one-hot
        out_colors = torch.zeros_like(x)
        
        # For each channel c in 0..9, if x[:, c, :, :] is 1, the new color is self.mapping[c]
        # We can just build the output one-hot directly!
        for c_in in range(10):
            c_out = int(self.mapping[c_in].item())
            out_colors[:, c_out:c_out+1, :, :] = torch.max(out_colors[:, c_out:c_out+1, :, :], x[:, c_in:c_in+1, :, :])
            
        return out_colors * valid_mask

def export_onnx():
    net = TaskNet()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/{task_id}.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )

if __name__ == '__main__':
    export_onnx()
