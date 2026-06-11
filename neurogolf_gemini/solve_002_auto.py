import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bg_color = 0
        self.fill_color = 4

    def forward(self, x):
        # x is (1, 10, H, W) one-hot
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
        
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
        border_connected = bg_layer * border_template
        
        # Iterative dilation using MaxPool2d (kernel 3, stride 1, padding 1)
        # MaxPool2d with padding 1 acts like morphological dilation!
        # But we want 4-connectivity (cross shape), so we can use a custom convolution.
        # However, for simplicity and since ARC grids are small, 3x3 square dilation is often fine if the holes 
        # don't have diagonal leakages. Wait, diagonal leakage MIGHT happen!
        # To strictly enforce 4-connectivity, we use a fixed depthwise conv2d kernel.
        
        cross_kernel = torch.tensor([[[[0., 1., 0.],
                                       [1., 1., 1.],
                                       [0., 1., 0.]]]], device=x.device)
                                       
        for _ in range(60): # Increase to 60 for large grids
            # Dilate
            dilated = F.conv2d(border_connected, cross_kernel, padding=1)
            # Clip to 1
            dilated = torch.clamp(dilated, 0.0, 1.0)
            # Mask with original bg_layer
            border_connected = dilated * bg_layer
            
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
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task002.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
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
