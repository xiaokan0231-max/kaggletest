import torch
import torch.nn as nn
import torch.nn.functional as F

class Task082(nn.Module):
    def forward(self, x):
        # x is [1, H, W], H is always 6
        row0 = x[:, 0:1, :] # [1, 1, W]
        row0_pad = F.pad(row0, (1, 1)) # [1, 1, W+2]
        
        shift_left = row0_pad[:, :, 2:]
        shift_right = row0_pad[:, :, :-2]
        
        row1 = torch.max(shift_left, shift_right)
        
        out = torch.cat([row0, row1, row0, row1, row0, row1], dim=1)
        return out

if __name__ == '__main__':
    model = Task082()
    dummy_input = torch.zeros((1, 6, 10), dtype=torch.int64)
    torch.onnx.export(
        model, 
        dummy_input, 
        '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task082.onnx',
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'width'}, 'output': {2: 'width'}}
    )
    print("ONNX exported successfully.")
