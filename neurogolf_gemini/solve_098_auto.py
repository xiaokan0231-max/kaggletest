import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os

class TaskNet(nn.Module):
    def forward(self, x):
        # x: (B, 10, H, W) one-hot encoded
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # For each non-zero pixel: if ALL 4 orthogonal neighbors have the same color, erase it.
        # "Same color" in one-hot means: the same channel is active.
        # 
        # For each color channel c (1..9):
        #   layer_c = x[:, c:c+1, :, :]
        #   A pixel at (r,c) is "interior" if layer_c[r,c]==1 AND all 4 neighbors also have layer_c==1
        #   This is equivalent to: eroded = min-pool of layer_c with 3x3 cross kernel
        #   interior = eroded (pixels where the entire 3x3 cross is 1)
        #   We erase these interior pixels.
        
        # Cross-shaped erosion kernel for conv2d
        # We use conv2d with a cross kernel and check if result == 5 (center + 4 neighbors)
        # Actually simpler: we want min of 5 positions. 
        # Min can be computed as: 1 - maxpool(1 - x)
        # Or: we can use conv2d with cross kernel and threshold at >= 5
        
        out = x.clone()
        
        for c in range(1, 10):
            layer = x[:, c:c+1, :, :]
            
            # Shift in 4 directions
            top_n = F.pad(layer[:, :, :-1, :], (0, 0, 1, 0))   # layer[r-1,c]
            bot_n = F.pad(layer[:, :, 1:, :], (0, 0, 0, 1))     # layer[r+1,c]
            left_n = F.pad(layer[:, :, :, :-1], (1, 0, 0, 0))   # layer[r,c-1]
            right_n = F.pad(layer[:, :, :, 1:], (0, 1, 0, 0))   # layer[r,c+1]
            
            # Interior = center AND all 4 neighbors
            interior = layer * top_n * bot_n * left_n * right_n
            
            # Erase interior: set color c to 0, set color 0 to 1
            out[:, c:c+1, :, :] = out[:, c:c+1, :, :] * (1.0 - interior)
            out[:, 0:1, :, :] = out[:, 0:1, :, :] + interior
        
        # Clamp to handle any potential overlap
        out = torch.clamp(out, 0.0, 1.0)
        
        return out * valid_mask

def export_onnx():
    net = TaskNet()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task098.onnx"
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
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False,
                    all_tensors_to_one_file=True, size_threshold=1024*1024*1024)
    if os.path.exists(onnx_path + '.data'):
        os.remove(onnx_path + '.data')

if __name__ == '__main__':
    export_onnx()
