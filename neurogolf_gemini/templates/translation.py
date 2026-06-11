import numpy as np

def fit_and_generate(task_data, task_id):
    """
    Tries to fit a simple translation template:
    Moves one specific color by (dx, dy), replacing vacated pixels with a background color.
    """
    if 'train' not in task_data:
        if isinstance(task_data, list):
            train_data = [d for d in task_data if 'output' in d]
        else:
            return None
    else:
        train_data = task_data['train']
        
    for ex in train_data:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        if inp.shape != out.shape:
            return None

    def test_translation(inp, out, target_color, dx, dy, bg_color):
        shifted = np.copy(inp)
        shifted[inp == target_color] = bg_color # erase
        
        ys, xs = np.where(inp == target_color)
        if len(ys) == 0:
            return np.array_equal(inp, out) # If color doesn't exist, must be equal
            
        new_ys = ys + dy
        new_xs = xs + dx
        
        if np.any(new_ys < 0) or np.any(new_ys >= inp.shape[0]) or np.any(new_xs < 0) or np.any(new_xs >= inp.shape[1]):
            return False
            
        shifted[new_ys, new_xs] = target_color
        return np.array_equal(shifted, out)

    # Search for (target_color, dx, dy, bg_color) that works for ALL train examples
    # The search space is small: color in 1..9, bg_color in 0..9, dx in -30..30, dy in -30..30
    best_params = None
    
    # Heuristics: find maximum possible dx, dy from the grid size to limit search
    H, W = np.array(train_data[0]['input']).shape
    
    for target_color in range(1, 10):
        for bg_color in range(0, 10):
            # To speed up, we can find the dx, dy from the first example directly!
            inp0 = np.array(train_data[0]['input'])
            out0 = np.array(train_data[0]['output'])
            y_in, x_in = np.where(inp0 == target_color)
            y_out, x_out = np.where(out0 == target_color)
            
            if len(y_in) == 0 or len(y_out) == 0 or len(y_in) != len(y_out):
                continue
                
            # If it's a pure translation, all pixels must shift by the same amount
            # Let's check the shift of the top-left-most pixel
            dy = y_out[0] - y_in[0]
            dx = x_out[0] - x_in[0]
            
            # Verify across all examples
            valid = True
            for ex in train_data:
                if not test_translation(np.array(ex['input']), np.array(ex['output']), target_color, dx, dy, bg_color):
                    valid = False
                    break
            
            if valid:
                best_params = (target_color, dx, dy, bg_color)
                break
        if best_params is not None:
            break
            
    if best_params is None:
        return None
        
    target_color, dx, dy, bg_color = best_params
    
    # Generate PyTorch code
    # We will use torch.roll to shift the target color channel, 
    # and then composite it back.
    
    code = f"""import torch
import torch.nn as nn
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.dx = {dx}
        self.dy = {dy}
        self.bg_color = {bg_color}

    def forward(self, x):
        # x is (1, 10, H, W) one-hot
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        
        # We need to shift target_layer by (dx, dy). Since ONNX doesn't support 
        # dynamic torch.roll well with dynamic sizes, we use padding and slicing if possible.
        # But wait, torch.roll IS supported in ONNX opset 14!
        # Actually, if dx and dy are dynamic? No, they are hardcoded.
        
        shifted_target = torch.roll(target_layer, shifts=(self.dy, self.dx), dims=(2, 3))
        
        # But wait! torch.roll wraps around! Translation in ARC usually DOES NOT wrap around.
        # So we must mask out the wrapped pixels.
        mask = torch.ones_like(target_layer)
        mask = torch.roll(mask, shifts=(self.dy, self.dx), dims=(2, 3))
        # Zero out the wrapped regions
        if self.dy > 0:
            mask[:, :, :self.dy, :] = 0
        elif self.dy < 0:
            mask[:, :, self.dy:, :] = 0
            
        if self.dx > 0:
            mask[:, :, :, :self.dx] = 0
        elif self.dx < 0:
            mask[:, :, :, self.dx:] = 0
            
        shifted_target = shifted_target * mask
        
        # Now composite:
        out_colors = x.clone()
        
        # 1. Erase target color from original position (replace with bg_color)
        out_colors[:, self.target_color:self.target_color+1, :, :] = 0
        out_colors[:, self.bg_color:self.bg_color+1, :, :] = torch.max(
            out_colors[:, self.bg_color:self.bg_color+1, :, :], 
            target_layer
        )
        
        # 2. Draw the shifted target
        # Replace whatever was there with target_color
        for c in range(10):
            if c != self.target_color:
                out_colors[:, c:c+1, :, :] = out_colors[:, c:c+1, :, :] * (1 - shifted_target)
                
        out_colors[:, self.target_color:self.target_color+1, :, :] = torch.max(
            out_colors[:, self.target_color:self.target_color+1, :, :],
            shifted_target
        )
            
        return out_colors * valid_mask

def export_onnx():
    net = TaskNet()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/{task_id}.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={{'input': {{0: 'batch_size', 2: 'height', 3: 'width'}},
                      'output': {{0: 'batch_size', 2: 'height', 3: 'width'}}}}
    )
    import os
    import onnx
    onnx_model = onnx.load_model(onnx_path, load_external_data=True)
    
    for tensor in onnx_model.graph.initializer:
        if tensor.HasField("data_location"):
            tensor.ClearField("data_location")
        if len(tensor.external_data) > 0:
            del tensor.external_data[:]
            
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                if attr.t.HasField("data_location"):
                    attr.t.ClearField("data_location")
                if len(attr.t.external_data) > 0:
                    del attr.t.external_data[:]
            for t in attr.tensors:
                if t.HasField("data_location"):
                    t.ClearField("data_location")
                if len(t.external_data) > 0:
                    del t.external_data[:]
                    
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        size_threshold=1024*1024*1024
    )
    if os.path.exists(onnx_path + '.data'):
        os.remove(onnx_path + '.data')

if __name__ == '__main__':
    export_onnx()
"""
    return code
