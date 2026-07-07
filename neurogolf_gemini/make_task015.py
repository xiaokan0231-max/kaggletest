import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Task015(nn.Module):
    def __init__(self):
        super().__init__()
        
    def shift(self, t, dy, dx):
        pad_t = max(0, dy)
        pad_b = max(0, -dy)
        pad_l = max(0, dx)
        pad_r = max(0, -dx)
        padded = F.pad(t, (pad_l, pad_r, pad_t, pad_b), value=0.0)
        return padded[:, :, pad_b:pad_b+t.size(2), pad_r:pad_r+t.size(3)]

    def forward(self, x):
        c2 = x[:, 2:3, :, :]
        c1 = x[:, 1:2, :, :]
        
        c4_add = torch.max(
            torch.max(self.shift(c2, -1, -1), self.shift(c2, -1, 1)),
            torch.max(self.shift(c2, 1, -1), self.shift(c2, 1, 1))
        )
        
        c7_add = torch.max(
            torch.max(self.shift(c1, -1, 0), self.shift(c1, 1, 0)),
            torch.max(self.shift(c1, 0, -1), self.shift(c1, 0, 1))
        )
        
        channels = []
        for i in range(10):
            if i == 0:
                c0 = F.relu(x[:, 0:1, :, :] - c4_add - c7_add)
                channels.append(c0)
            elif i == 4:
                channels.append(torch.max(x[:, 4:5, :, :], c4_add))
            elif i == 7:
                channels.append(torch.max(x[:, 7:8, :, :], c7_add))
            else:
                channels.append(x[:, i:i+1, :, :])
                
        return torch.cat(channels, dim=1)

def export_and_test():
    model = Task015()
    model.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "neurogolf", "data", "working", "task015.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    import sys
    utils_path = os.path.join(base_dir, 'neurogolf', 'data', 'raw', 'neurogolf_utils')
    sys.path.insert(0, utils_path)
    import neurogolf_utils
    
    neurogolf_utils._NEUROGOLF_DIR = os.path.join(base_dir, 'neurogolf', 'data', 'raw') + '/'
    
    examples = neurogolf_utils.load_examples(15)
    import onnx
    model_proto = onnx.load(onnx_path)
    neurogolf_utils.verify_network(model_proto, 15, examples)

if __name__ == "__main__":
    export_and_test()
