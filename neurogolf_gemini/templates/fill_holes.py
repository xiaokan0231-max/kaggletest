import numpy as np

def connected_to_border(mask):
    """
    Given a boolean mask, returns a mask of the same size with True for all pixels
    that are connected to the border (4-connectivity).
    """
    import scipy.ndimage
    # 4-connectivity structure
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    labeled, _ = scipy.ndimage.label(mask, structure=structure)
    
    border_labels = set()
    if mask.shape[0] > 0 and mask.shape[1] > 0:
        border_labels.update(labeled[0, :])
        border_labels.update(labeled[-1, :])
        border_labels.update(labeled[:, 0])
        border_labels.update(labeled[:, -1])
        
    border_labels.discard(0) # 0 is background in label()
    
    border_connected = np.isin(labeled, list(border_labels))
    return border_connected

def fit_and_generate(task_data, task_id):
    """
    Tries to fit a 'fill holes' template:
    Finds regions of bg_color that are fully enclosed by other colors (not connected to border),
    and fills them with fill_color.
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

    # Search for (bg_color, fill_color)
    best_params = None
    
    for bg_color in range(10):
        for fill_color in range(10):
            if bg_color == fill_color:
                continue
                
            valid = True
            for ex in train_data:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                
                # compute holes
                bg_mask = (inp == bg_color)
                border_connected = connected_to_border(bg_mask)
                hole_mask = bg_mask & (~border_connected)
                
                expected_out = np.copy(inp)
                expected_out[hole_mask] = fill_color
                
                if not np.array_equal(expected_out, out):
                    valid = False
                    break
                    
            if valid:
                best_params = (bg_color, fill_color)
                break
        if best_params is not None:
            break
            
    if best_params is None:
        return None
        
    bg_color, fill_color = best_params
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg_color = {bg_color}
        self.fill_color = {fill_color}

    def forward(self, x):
        # x is (1, 10, H, W) one-hot
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        bg_layer_full = bg_layer + (1.0 - valid_mask)
        
        # Initialize border mask
        B, C, H, W = bg_layer.shape
        border_mask = torch.zeros_like(bg_layer)
        # We need to set the edges to 1 if they are bg_color
        # Because ONNX does not support dynamic slice assignment well, we can create a border template
        # by taking a ones tensor and zeroing the inside
        ones = torch.ones_like(bg_layer)
        inside = F.pad(torch.zeros(B, C, H-2, W-2, device=x.device), pad=(1,1,1,1), value=1.0)
        # inside is 0 on the border and 1 inside? No, F.pad pads with 1.0! So border is 1.0, inside is 0.0.
        border_template = inside
        
        # Initial border connected components
        border_connected = bg_layer_full * border_template
        
        # Iterative dilation using MaxPool2d (kernel 3, stride 1, padding 1)
        # MaxPool2d with padding 1 acts like morphological dilation!
        # But we want 4-connectivity (cross shape), so we can use a custom convolution.
        # However, for simplicity and since ARC grids are small, 3x3 square dilation is often fine if the holes 
        # don't have diagonal leakages. Wait, diagonal leakage MIGHT happen!
        # To strictly enforce 4-connectivity, we use a fixed depthwise conv2d kernel.
        
        cross_kernel = torch.tensor([[[[0., 1., 0.],
                                       [1., 1., 1.],
                                       [0., 1., 0.]]]], device=x.device)
                                       
        for _ in range(60): # 60 iterations to cover 30x30 grid
            # Dilate
            dilated = F.conv2d(border_connected, cross_kernel, padding=1)
            # Clip to 1
            dilated = torch.clamp(dilated, 0.0, 1.0)
            # Mask with original bg_layer_full so it can travel through padding
            border_connected = dilated * bg_layer_full
            
        # Holes are bg_layer pixels that are NOT in border_connected
        # Since tensors are 0 or 1 float:
        holes = bg_layer * (1.0 - border_connected)
        
        # Now construct output
        out_colors = x.clone()
        # Remove holes from bg_color channel
        out_colors[:, self.bg_color:self.bg_color+1, :, :] = out_colors[:, self.bg_color:self.bg_color+1, :, :] * (1.0 - holes)
        # Add holes to fill_color channel
        out_colors[:, self.fill_color:self.fill_color+1, :, :] = torch.max(
            out_colors[:, self.fill_color:self.fill_color+1, :, :],
            holes
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
