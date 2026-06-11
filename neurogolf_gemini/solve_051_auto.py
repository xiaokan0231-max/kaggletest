import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Task051(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        counts = x.sum(dim=(2,3), keepdim=True)
        color_idx = torch.arange(10, dtype=x.dtype, device=x.device).view(1, 10, 1, 1)
        
        is_antenna_color = (counts == 1) & (color_idx > 0)
        antenna_mask = (x * is_antenna_color.float()).sum(dim=1, keepdim=True)
        
        is_dish_color = (counts > 1) & (color_idx > 0)
        dish_mask = (x * is_dish_color.float()).sum(dim=1, keepdim=True)
        
        shifted_up = F.pad(antenna_mask, (0, 0, 0, 1))[:, :, 1:, :]
        shifted_down = F.pad(antenna_mask, (0, 0, 1, 0))[:, :, :-1, :]
        shifted_left = F.pad(antenna_mask, (0, 1, 0, 0))[:, :, :, 1:]
        shifted_right = F.pad(antenna_mask, (1, 0, 0, 0))[:, :, :, :-1]
        
        has_dish_up = (dish_mask * shifted_up).sum(dim=(2,3), keepdim=True)
        has_dish_down = (dish_mask * shifted_down).sum(dim=(2,3), keepdim=True)
        has_dish_left = (dish_mask * shifted_left).sum(dim=(2,3), keepdim=True)
        has_dish_right = (dish_mask * shifted_right).sum(dim=(2,3), keepdim=True)
        
        X_dir = has_dish_right - has_dish_left
        Y_dir = has_dish_down - has_dish_up
        
        X_pos = (X_dir > 0).float()
        X_neg = (X_dir < 0).float()
        Y_pos = (Y_dir > 0).float()
        Y_neg = (Y_dir < 0).float()
        
        current = antenna_mask
        ray = torch.zeros_like(current)
        
        for _ in range(35):
            ray = torch.max(ray, current)
            c_right = F.pad(current, (1, 0, 0, 0))[:, :, :, :-1]
            c_left = F.pad(current, (0, 1, 0, 0))[:, :, :, 1:]
            c_down = F.pad(current, (0, 0, 1, 0))[:, :, :-1, :]
            c_up = F.pad(current, (0, 0, 0, 1))[:, :, 1:, :]
            current = X_pos * c_right + X_neg * c_left + Y_pos * c_down + Y_neg * c_up

        bg_mask = x[:, 0:1, :, :]
        apply_ray = ray * bg_mask
        
        antenna_color_one_hot = is_antenna_color.float()
        
        one_hot_0 = torch.zeros(1, 10, 1, 1, device=x.device, dtype=x.dtype)
        one_hot_0[:, 0, :, :] = 1.0
        
        out = x + apply_ray * antenna_color_one_hot - apply_ray * one_hot_0
        return out

def export_and_test():
    model = Task051()
    model.eval()
    
    # Needs to be able to handle variable sizes but opset 14 allows dynamic_axes,
    # however, we are asked to not use NONZERO, LOOP, SCAN, UNIQUE, or dynamic shape repeats
    # We will export with a static dummy input size like 30x30, but allow dynamic axes?
    # Let's check how the script was exporting before: dynamic_axes=None!
    # Wait, the prompt says: "use padded constant-size Tensors (e.g. 30x30 max, or use x.shape but keep ops static like F.conv2d, torch.roll, etc.)".
    # Since we used `x.sum(dim=(2,3))` and `F.pad`, those are shape-agnostic!
    # Let's export with dynamic axes to be safe and compatible with variable inputs, or 30x30.
    # The prompt actually says: "use padded constant-size Tensors (e.g. 30x30 max, or use x.shape but keep ops static like F.conv2d, torch.roll, etc.)".
    # Let's just use dynamic_axes to allow any shape, because our operations are independent of spatial shape.
    
    dummy_input = torch.zeros(1, 10, 15, 15)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "neurogolf", "data", "working", "task051.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    
    utils_path = os.path.join(base_dir, 'neurogolf_gemini')
    sys.path.insert(0, utils_path)
    import neurogolf_utils
    
    neurogolf_utils._NEUROGOLF_DIR = os.path.join(base_dir, 'neurogolf', 'data', 'raw') + '/'
    
    examples = neurogolf_utils.load_examples(51)
    import onnx
    model_proto = onnx.load(onnx_path)
    neurogolf_utils.verify_network(model_proto, 51, examples)

if __name__ == "__main__":
    export_and_test()
