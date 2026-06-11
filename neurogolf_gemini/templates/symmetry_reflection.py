import numpy as np

def fit_and_generate(task_data, task_id):
    """
    Tries to fit a symmetry reflection template.
    Mirrors a target_color either horizontally or vertically (with an optional shift),
    and stamps it onto the background.
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

    # Possible flips: 1 (vertical flip, up-down), 2 (horizontal flip, left-right)
    flips = [
        ('V', lambda x: np.flip(x, axis=0)),
        ('H', lambda x: np.flip(x, axis=1)),
        ('VH', lambda x: np.flip(np.flip(x, axis=0), axis=1))
    ]
    
    best_params = None
    
    # We allow the flipped image to be shifted by (dy, dx)
    # The search space is: target_color, bg_color, flip_type, dy, dx
    
    # Heuristics: to find dy, dx, we can just look at the difference between output and input
    # output - input = new pixels added.
    # The new pixels MUST be a shifted and flipped version of some pixels in the input.
    
    for ex0 in [train_data[0]]:
        inp0 = np.array(ex0['input'])
        out0 = np.array(ex0['output'])
        
        for target_color in range(1, 10):
            for bg_color in range(0, 10):
                if target_color == bg_color:
                    continue
                    
                target_mask = (inp0 == target_color)
                if not np.any(target_mask):
                    continue
                    
                # The "added" pixels of target_color in the output
                added_mask = (out0 == target_color) & (inp0 != target_color)
                
                if not np.any(added_mask):
                    continue
                    
                ys_added, xs_added = np.where(added_mask)
                
                for flip_name, flip_func in flips:
                    flipped_target = flip_func(target_mask)
                    ys_flipped, xs_flipped = np.where(flipped_target)
                    
                    if len(ys_added) == 0 or len(ys_flipped) == 0:
                        continue
                        
                    # Calculate required shift from the first pixel
                    # This assumes the first pixel in top-bottom left-right scan matches
                    dy = ys_added[0] - ys_flipped[0]
                    dx = xs_added[0] - xs_flipped[0]
                    
                    # Verify across all examples
                    valid = True
                    for ex in train_data:
                        inp = np.array(ex['input'])
                        out = np.array(ex['output'])
                        
                        t_mask = (inp == target_color)
                        f_mask = flip_func(t_mask)
                        
                        expected_out = np.copy(inp)
                        
                        # Apply shift
                        ys, xs = np.where(f_mask)
                        new_ys = ys + dy
                        new_xs = xs + dx
                        
                        # Filter out of bounds
                        valid_idx = (new_ys >= 0) & (new_ys < inp.shape[0]) & (new_xs >= 0) & (new_xs < inp.shape[1])
                        new_ys = new_ys[valid_idx]
                        new_xs = new_xs[valid_idx]
                        
                        # Only overwrite bg_color
                        overwrite_idx = inp[new_ys, new_xs] == bg_color
                        new_ys = new_ys[overwrite_idx]
                        new_xs = new_xs[overwrite_idx]
                        
                        expected_out[new_ys, new_xs] = target_color
                        
                        if not np.array_equal(expected_out, out):
                            valid = False
                            break
                            
                    if valid:
                        best_params = (target_color, bg_color, flip_name, dy, dx)
                        break
                if best_params: break
            if best_params: break
            
    if not best_params:
        return None
        
    target_color, bg_color, flip_name, dy, dx = best_params
    
    flip_dims = []
    if 'V' in flip_name:
        flip_dims.append(2)
    if 'H' in flip_name:
        flip_dims.append(3)
        
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.bg_color = {bg_color}
        self.flip_dims = {flip_dims}
        self.dy = {dy}
        self.dx = {dx}

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        
        flipped = torch.flip(target_layer, dims=self.flip_dims)
        
        # Shift flipped layer by (dy, dx) using roll and mask
        shifted = torch.roll(flipped, shifts=(self.dy, self.dx), dims=(2, 3))
        
        mask = torch.ones_like(shifted)
        mask = torch.roll(mask, shifts=(self.dy, self.dx), dims=(2, 3))
        if self.dy > 0:
            mask[:, :, :self.dy, :] = 0
        elif self.dy < 0:
            mask[:, :, self.dy:, :] = 0
            
        if self.dx > 0:
            mask[:, :, :, :self.dx] = 0
        elif self.dx < 0:
            mask[:, :, :, self.dx:] = 0
            
        shifted = shifted * mask
        
        # Only overwrite bg_color
        to_add = shifted * bg_layer
        
        out_colors = x.clone()
        out_colors[:, self.bg_color:self.bg_color+1, :, :] = out_colors[:, self.bg_color:self.bg_color+1, :, :] * (1.0 - to_add)
        out_colors[:, self.target_color:self.target_color+1, :, :] = torch.max(
            out_colors[:, self.target_color:self.target_color+1, :, :],
            to_add
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
