import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Task016(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: [1, 10, 30, 30]
        c0 = x[:, 0:1, :, :]
        c1 = x[:, 5:6, :, :]
        c2 = x[:, 6:7, :, :]
        c3 = x[:, 4:5, :, :]
        c4 = x[:, 3:4, :, :]
        c5 = x[:, 1:2, :, :]
        c6 = x[:, 2:3, :, :]
        c7 = x[:, 7:8, :, :]
        c8 = x[:, 9:10, :, :]
        c9 = x[:, 8:9, :, :]
        
        return torch.cat([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)

def export_and_test():
    model = Task016()
    model.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "neurogolf", "data", "working", "task016.onnx")
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
    
    examples = neurogolf_utils.load_examples(16)
    import onnx
    model_proto = onnx.load(onnx_path)
    neurogolf_utils.verify_network(model_proto, 16, examples)

if __name__ == "__main__":
    export_and_test()
