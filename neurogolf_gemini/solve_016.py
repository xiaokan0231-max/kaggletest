import torch
import torch.nn as nn
import onnx

class Task016Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        out_channels = [
            x[:, 0:1, :, :], # 0 -> 0
            x[:, 5:6, :, :], # 1 -> 5
            x[:, 6:7, :, :], # 2 -> 6
            x[:, 4:5, :, :], # 3 -> 4
            x[:, 3:4, :, :], # 4 -> 3
            x[:, 1:2, :, :], # 5 -> 1
            x[:, 2:3, :, :], # 6 -> 2
            x[:, 7:8, :, :], # 7 -> 7
            x[:, 9:10, :, :], # 8 -> 9
            x[:, 8:9, :, :], # 9 -> 8
        ]
        
        out = torch.cat(out_channels, dim=1)
        
        # Apply valid mask just in case
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task016Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task016.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}},
        dynamo=False
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
