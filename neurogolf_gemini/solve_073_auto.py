import torch
import torch.nn as nn
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = 1
        self.dx = 0
        self.dy = 2
        self.bg_color = 0

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
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task073.onnx"
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
