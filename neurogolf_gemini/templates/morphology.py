import numpy as np
import scipy.ndimage

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

    kernels = {
        'cross': np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=bool),
        'square': np.array([[1,1,1], [1,1,1], [1,1,1]], dtype=bool),
        'horizontal': np.array([[0,0,0], [1,1,1], [0,0,0]], dtype=bool),
        'vertical': np.array([[0,1,0], [0,1,0], [0,1,0]], dtype=bool),
        'diag_tl_br': np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=bool),
        'diag_tr_bl': np.array([[0,0,1], [0,1,0], [1,0,0]], dtype=bool),
    }

    best_params = None
    
    for target_color in range(10):
        for out_color in range(10):
            for k_name, kernel in kernels.items():
                for steps in range(1, 6):
                    for keep_original in [True, False]:
                        valid = True
                        for ex in train_data:
                            inp = np.array(ex['input'])
                            out = np.array(ex['output'])
                            
                            mask = (inp == target_color)
                            
                            for _ in range(steps):
                                mask = scipy.ndimage.binary_dilation(mask, structure=kernel)
                                
                            expected_out = np.copy(inp)
                            if not keep_original:
                                expected_out[inp == target_color] = 0 # assuming bg is 0
                                
                            # We only overwrite where the new mask is true
                            expected_out[mask] = out_color
                            
                            if keep_original:
                                # target_color pixels should remain target_color
                                expected_out[inp == target_color] = target_color
                                
                            if not np.array_equal(expected_out, out):
                                valid = False
                                break
                                
                        if valid:
                            best_params = (target_color, out_color, k_name, steps, keep_original)
                            break
                    if best_params: break
                if best_params: break
            if best_params: break
        if best_params: break
        
    if best_params is None:
        return None
        
    target_color, out_color, k_name, steps, keep_original = best_params
    
    # Generate ONNX code
    kernel_tensor_str = "[" + ", ".join(["[" + ", ".join([str(float(v)) for v in row]) + "]" for row in kernels[k_name]]) + "]"
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.out_color = {out_color}
        self.steps = {steps}
        self.keep_original = {keep_original}

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        B, C, H, W = x.shape
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        
        kernel = torch.tensor([[{kernel_tensor_str}]], device=x.device)
        
        dilated = target_layer
        for _ in range(self.steps):
            dilated = F.conv2d(dilated, kernel, padding=1)
            dilated = torch.clamp(dilated, 0.0, 1.0)
            
        out_colors = x.clone()
        
        # Remove old colors where dilated
        out_colors = out_colors * (1.0 - dilated)
        
        # Add out_color where dilated
        out_colors[:, self.out_color:self.out_color+1, :, :] = torch.max(
            out_colors[:, self.out_color:self.out_color+1, :, :],
            dilated
        )
        
        # Ensure keep_original overwrites the dilated mask for the target_color
        if self.keep_original:
            out_colors = out_colors * (1.0 - target_layer)
            out_colors[:, self.target_color:self.target_color+1, :, :] = torch.max(
                out_colors[:, self.target_color:self.target_color+1, :, :],
                target_layer
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
