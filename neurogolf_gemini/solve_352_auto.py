import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = 2
        self.out_color = 1
        self.steps = 1
        self.keep_original = True

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        B, C, H, W = x.shape
        
        target_layer = x[:, self.target_color:self.target_color+1, :, :]
        
        kernel = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=x.device)
        
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
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task352.onnx"
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
