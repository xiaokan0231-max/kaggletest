import numpy as np

def fit_and_generate(task_data, task_id):
    """
    Tries to fit a gravity drop template.
    Moves a target_color in a specific direction (dr, dc) as far as possible 
    until it hits an obstacle (any color other than bg_color and target_color) or the boundary.
    All pixels of target_color move rigidly together.
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

    dirs = {
        'S': (1, 0), 'N': (-1, 0), 'E': (0, 1), 'W': (0, -1)
    }
    
    best_params = None
    
    for target_color in range(1, 10):
        for bg_color in range(0, 10):
            if target_color == bg_color:
                continue
                
            for dname, (dr, dc) in dirs.items():
                valid = True
                for ex in train_data:
                    inp = np.array(ex['input'])
                    out = np.array(ex['output'])
                    
                    target_mask = (inp == target_color)
                    if not np.any(target_mask):
                        # If target_color is not present, output must equal input
                        if not np.array_equal(inp, out):
                            valid = False
                            break
                        continue
                        
                    obstacle_mask = (inp != bg_color) & (inp != target_color)
                    
                    ys, xs = np.where(target_mask)
                    
                    max_k = 0
                    for k in range(1, 31):
                        new_ys = ys + k * dr
                        new_xs = xs + k * dc
                        
                        # Check bounds
                        if np.any(new_ys < 0) or np.any(new_ys >= inp.shape[0]) or \
                           np.any(new_xs < 0) or np.any(new_xs >= inp.shape[1]):
                            break
                            
                        # Check obstacles
                        if np.any(obstacle_mask[new_ys, new_xs]):
                            break
                            
                        max_k = k
                        
                    # Now apply the shift
                    expected_out = np.copy(inp)
                    expected_out[target_mask] = bg_color
                    new_ys = ys + max_k * dr
                    new_xs = xs + max_k * dc
                    expected_out[new_ys, new_xs] = target_color
                    
                    if not np.array_equal(expected_out, out):
                        valid = False
                        break
                        
                if valid:
                    best_params = (target_color, bg_color, dr, dc)
                    break
            if best_params:
                break
        if best_params:
            break
            
    if not best_params:
        return None
        
    target_color, bg_color, dr, dc = best_params
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.bg_color = {bg_color}
        self.dr = {dr}
        self.dc = {dc}

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        
        obstacle_layer = valid_mask - target_layer - bg_layer
        
        B, C, H, W = x.shape
        
        # We need to find the maximum shift k such that target shifted by k does not overlap obstacle or go out of bounds.
        # Since max grid size is 30, we evaluate 30 shifts.
        
        # To handle bounds, we can pad the obstacle_layer with 1s on the boundary!
        # Instead of explicitly padding, shifting target layer out of bounds will drop those pixels.
        # We can detect out of bounds by checking if the sum of pixels in shifted target equals the sum in original target.
        # But wait! sum() across spatial dims might be dynamic? Yes, ONNX supports reduce_sum.
        
        target_sum = torch.sum(target_layer, dim=(2, 3), keepdim=True)
        
        max_k = torch.zeros((B, 1, 1, 1), device=x.device, dtype=torch.float32)
        valid_so_far = torch.ones((B, 1, 1, 1), device=x.device, dtype=torch.float32)
        
        # We will accumulate the best shifted_target
        best_shifted = target_layer.clone()
        
        current_shifted = target_layer
        
        for k in range(1, 31):
            # Shift by 1 step in (dr, dc)
            # Use padding to shift:
            # pad format is (left, right, top, bottom)
            # to shift right (dc=1), pad left 1, right -1 -> actually pad(1, 0, 0, 0) and slice
            
            pad_left = max(0, self.dc)
            pad_right = max(0, -self.dc)
            pad_top = max(0, self.dr)
            pad_bottom = max(0, -self.dr)
            
            # Pad with 0
            padded = F.pad(current_shifted, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            
            # Slice back to original size
            if self.dr > 0:
                padded = padded[:, :, :-self.dr, :]
            elif self.dr < 0:
                padded = padded[:, :, -self.dr:, :]
                
            if self.dc > 0:
                padded = padded[:, :, :, :-self.dc]
            elif self.dc < 0:
                padded = padded[:, :, :, -self.dc:]
                
            current_shifted = padded
            
            # Check conditions:
            # 1. No pixels lost (out of bounds)
            current_sum = torch.sum(current_shifted, dim=(2, 3), keepdim=True)
            # using a tolerance for float comparison
            in_bounds = (current_sum > target_sum - 0.5).float()
            
            # 2. No overlap with obstacle
            overlap = torch.sum(current_shifted * obstacle_layer, dim=(2, 3), keepdim=True)
            no_overlap = (overlap < 0.5).float()
            
            is_valid = in_bounds * no_overlap
            valid_so_far = valid_so_far * is_valid
            
            # If valid_so_far is 1, we update best_shifted
            # We can't use dynamic indexing easily, so we interpolate
            best_shifted = valid_so_far * current_shifted + (1.0 - valid_so_far) * best_shifted
            
        # Composite output
        out_colors = x.clone()
        # Erase target
        out_colors[:, self.target_color:self.target_color+1, :, :] = 0
        out_colors[:, self.bg_color:self.bg_color+1, :, :] = torch.max(
            out_colors[:, self.bg_color:self.bg_color+1, :, :],
            target_layer
        )
        
        # Draw shifted target
        # Anything where best_shifted is 1 becomes target_color, overwriting bg
        out_colors[:, self.bg_color:self.bg_color+1, :, :] = out_colors[:, self.bg_color:self.bg_color+1, :, :] * (1.0 - best_shifted)
        out_colors[:, self.target_color:self.target_color+1, :, :] = torch.max(
            out_colors[:, self.target_color:self.target_color+1, :, :],
            best_shifted
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
