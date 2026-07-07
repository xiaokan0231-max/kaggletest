import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Task007(nn.Module):
    def __init__(self):
        super().__init__()
        
    def shift(self, t, dy, dx):
        pad_t = max(0, dy)
        pad_b = max(0, -dy)
        pad_l = max(0, dx)
        pad_r = max(0, -dx)
        padded = F.pad(t, (pad_l, pad_r, pad_t, pad_b), value=0.0)
        
        y_start = pad_b
        y_end = -pad_t if pad_t > 0 else None
        
        x_start = pad_r
        x_end = -pad_l if pad_l > 0 else None
        
        return padded[:, :, y_start:y_end, x_start:x_end]

    def forward(self, x):
        # We want to flood fill using shifts that preserve (x+y)%3:
        # dx=1, dy=-1 (anti-diagonal)
        # dx=-1, dy=1 (anti-diagonal)
        # dx=3, dy=0 (horizontal period 3)
        # dx=-3, dy=0 (horizontal period 3)
        # dx=0, dy=3 (vertical period 3)
        # dx=0, dy=-3 (vertical period 3)
        
        out = x.clone()
        out[:, 0, :, :] = 0.0
        
        # Max grid dimension is 30. So 15 iterations of shifts will easily cover the whole grid.
        for _ in range(15):
            s1 = self.shift(out, 1, -1)
            s2 = self.shift(out, -1, 1)
            s3 = self.shift(out, 3, 0)
            s4 = self.shift(out, -3, 0)
            s5 = self.shift(out, 0, 3)
            s6 = self.shift(out, 0, -3)
            
            out = torch.max(out, s1)
            out = torch.max(out, s2)
            out = torch.max(out, s3)
            out = torch.max(out, s4)
            out = torch.max(out, s5)
            out = torch.max(out, s6)
            
        # Mask out the padding area
        valid_mask = torch.sum(x, dim=1, keepdim=True)
        out = out * valid_mask
            
        return out

def export_and_test():
    model = Task007()
    model.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "neurogolf", "data", "working", "task007.onnx")
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
    
    examples = neurogolf_utils.load_examples(7)
    import onnx
    model_proto = onnx.load(onnx_path)
    neurogolf_utils.verify_network(model_proto, 7, examples)

if __name__ == "__main__":
    export_and_test()
