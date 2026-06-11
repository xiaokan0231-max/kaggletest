import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sep = x[:, :, 3:4, 3:4]
        is_sep = (x == sep)
        is_zero = (x == 0)
        is_active = ~(is_sep | is_zero)
        
        x_active = x * is_active.long()
        color = F.max_pool2d(x_active.float(), kernel_size=11).long()
        
        counts = F.avg_pool2d(is_active.float(), kernel_size=3, stride=4, padding=0) * 9.0
        counts = torch.round(counts)
        
        max_count = F.max_pool2d(counts, kernel_size=3)
        winning = (counts == max_count).long()
        
        w_up = winning.unsqueeze(3).unsqueeze(5)
        w_up = w_up.expand(-1, -1, 3, 4, 3, 4)
        B = x.shape[0]
        w_up = w_up.reshape(B, 1, 12, 12)
        w_up = w_up[:, :, :11, :11]
        
        out = w_up * color * (~is_sep).long() + sep * is_sep.long()
        return out

if __name__ == '__main__':
    os.makedirs('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working', exist_ok=True)
    model = Model()
    model.eval()
    dummy_input = torch.zeros(1, 1, 11, 11, dtype=torch.int64)
    # Using TorchScript exporter explicitly if possible, else standard export
    torch.onnx.export(
        model, 
        dummy_input, 
        '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task059.onnx',
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model exported to /Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task059.onnx")
