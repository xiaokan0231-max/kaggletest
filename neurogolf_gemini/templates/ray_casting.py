import numpy as np

def fit_and_generate(task_data, task_id):
    if 'train' not in task_data:
        if isinstance(task_data, list):
            train_data = [d for d in task_data if 'output' in d]
        else:
            return None
    else:
        train_data = task_data['train']
        
    for ex in train_data:
        if np.array(ex['input']).shape != np.array(ex['output']).shape:
            return None

    dirs = {
        'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1),
        'NE': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1), 'SW': (1, -1)
    }
    
    bg_color = 0
    best_params = None
    
    for target_color in range(10):
        for ray_color in range(10):
            valid_dirs = set(dirs.keys())
            
            # Step 1: Filter out invalid directions
            for ex in train_data:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                H, W = inp.shape
                
                rays = {}
                for dname, (dr, dc) in dirs.items():
                    ray_mask = np.zeros_like(inp, dtype=bool)
                    for r in range(H):
                        for c in range(W):
                            if inp[r, c] == target_color:
                                cr, cc = r + dr, c + dc
                                while 0 <= cr < H and 0 <= cc < W and inp[cr, cc] == bg_color:
                                    ray_mask[cr, cc] = True
                                    cr += dr
                                    cc += dc
                    rays[dname] = ray_mask
                    
                for dname in list(valid_dirs):
                    # If this direction puts a ray where output is NOT ray_color (and not target_color, in case it overwrites target)
                    # Actually, we expect output to be exactly ray_color there
                    invalid_pixels = rays[dname] & (out != ray_color) & (out != target_color)
                    if np.any(invalid_pixels):
                        valid_dirs.remove(dname)
                        
            # Step 2: Check if valid_dirs exactly reconstructs the output
            if not valid_dirs:
                continue
                
            match = True
            for ex in train_data:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                H, W = inp.shape
                
                expected_out = np.copy(inp)
                for dname in valid_dirs:
                    dr, dc = dirs[dname]
                    for r in range(H):
                        for c in range(W):
                            if inp[r, c] == target_color:
                                cr, cc = r + dr, c + dc
                                while 0 <= cr < H and 0 <= cc < W and inp[cr, cc] == bg_color:
                                    expected_out[cr, cc] = ray_color
                                    cr += dr
                                    cc += dc
                
                if not np.array_equal(expected_out, out):
                    match = False
                    break
                    
            if match:
                best_params = (target_color, ray_color, sorted(list(valid_dirs)))
                break
        if best_params:
            break
            
    if not best_params:
        return None
        
    target_color, ray_color, chosen_dirs = best_params
    
    # Kernel matrices for ONNX F.conv2d
    # To shift a mask in direction (dr, dc), we convolve with a 3x3 kernel that has 1 at (-dr+1, -dc+1)
    kernels_str = []
    for dname in chosen_dirs:
        dr, dc = dirs[dname]
        kr, kc = -dr + 1, -dc + 1
        k = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        k[kr][kc] = 1.0
        kernels_str.append(f"[{k[0]}, {k[1]}, {k[2]}]")
        
    kernels_tensor_str = "[" + ", ".join(["[" + k + "]" for k in kernels_str]) + "]"
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.ray_color = {ray_color}
        self.bg_color = {bg_color}

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        B, C, H, W = x.shape
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        
        # We have {len(chosen_dirs)} directions
        kernels = torch.tensor({kernels_tensor_str}, device=x.device).unsqueeze(1) # shape (N, 1, 3, 3)
        
        cumulative_ray = torch.zeros_like(target_layer)
        
        for i in range(kernels.shape[0]):
            k = kernels[i:i+1] # shape (1, 1, 3, 3)
            ray = target_layer
            # Propagate up to 30 steps
            for _ in range(30):
                ray = F.conv2d(ray, k, padding=1)
                ray = torch.clamp(ray, 0.0, 1.0)
                ray = ray * bg_layer
                cumulative_ray = torch.max(cumulative_ray, ray)
                
        out_colors = x.clone()
        
        # Remove background where ray is present
        out_colors = out_colors * (1.0 - cumulative_ray)
        
        # Add ray color
        out_colors[:, self.ray_color:self.ray_color+1, :, :] = torch.max(
            out_colors[:, self.ray_color:self.ray_color+1, :, :],
            cumulative_ray
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
