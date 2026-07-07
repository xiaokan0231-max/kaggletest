import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task002(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        valid_mask = torch.sum(x, dim=1, keepdim=True)
        boundary = x[:, 3:4, :, :]
        effective_boundary = torch.max(boundary, 1.0 - valid_mask)

        valid_left = F.pad(valid_mask[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        right_edge = F.relu(valid_mask - valid_left)
        valid_up = F.pad(valid_mask[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
        bottom_edge = F.relu(valid_mask - valid_up)

        top_edge = F.pad(valid_mask[:, :, 1:, :] * 0.0, (0, 0, 1, 0), value=1.0) * valid_mask
        left_edge = F.pad(valid_mask[:, :, :, 1:] * 0.0, (1, 0, 0, 0), value=1.0) * valid_mask

        state = torch.max(torch.max(right_edge, bottom_edge), torch.max(top_edge, left_edge))
        state = F.relu(state - effective_boundary)

        for _ in range(30):
            dilate_h = F.max_pool2d(state, kernel_size=(1,3), stride=1, padding=(0,1))
            dilate_v = F.max_pool2d(state, kernel_size=(3,1), stride=1, padding=(1,0))
            state = torch.max(dilate_h, dilate_v)
            state = F.relu(state - effective_boundary)

        inside = F.relu(valid_mask - torch.max(state, boundary))

        channels = []
        for i in range(10):
            if i == 4:
                channels.append(torch.max(x[:, 4:5, :, :], inside))
            else:
                channels.append(F.relu(x[:, i:i+1, :, :] - inside))
                
        return torch.cat(channels, dim=1)

def export_and_test():
    model = Task002()
    model.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "neurogolf", "data", "working", "task002.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    print(f"Exported to {onnx_path}")
    
    # Verify using neurogolf_utils
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    utils_path = os.path.join(base_dir, 'neurogolf', 'data', 'raw', 'neurogolf_utils')
    sys.path.insert(0, utils_path)
    import neurogolf_utils
    
    neurogolf_utils._NEUROGOLF_DIR = os.path.join(base_dir, 'neurogolf', 'data', 'raw') + '/'
    
    examples = neurogolf_utils.load_examples(2)
    import onnx
    model_proto = onnx.load(onnx_path)
    neurogolf_utils.verify_network(model_proto, 2, examples)

if __name__ == "__main__":
    export_and_test()
