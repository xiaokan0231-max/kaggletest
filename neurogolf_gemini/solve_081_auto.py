import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os

class TaskNet(nn.Module):
    def forward(self, x):
        # x: (B, 10, H, W) one-hot encoded
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # Get layer for color 8
        c8 = x[:, 8:9, :, :]  # (B, 1, H, W)
        # Get layer for color 0 (background)
        c0 = x[:, 0:1, :, :]  # (B, 1, H, W)
        
        # Detect L-shaped corners: a 0-pixel with exactly 2 orthogonal 8-neighbors
        # that form an L (i.e., adjacent pair, not opposite pair).
        # 
        # 4 possible L corners at position (r,c):
        # top-left:  c8[r-1,c]*c8[r,c-1]
        # top-right: c8[r-1,c]*c8[r,c+1]
        # bot-left:  c8[r+1,c]*c8[r,c-1]
        # bot-right: c8[r+1,c]*c8[r,c+1]
        
        # Use convolution kernels to detect these patterns
        # Kernel for top-left corner: detects if (r-1,c) and (r,c-1) are both 8
        # We convolve c8 with a 3x3 kernel and check if the result >= 2
        
        # Actually, simpler: shift c8 in 4 directions and multiply pairs
        # top = roll up
        top = F.pad(c8[:, :, :-1, :], (0, 0, 1, 0))  # shift down = pad top
        # Actually let me think about this more carefully.
        # I want: top_neighbor[r,c] = c8[r-1,c]
        # That means: shift c8 down by 1 → top_neighbor = c8 shifted down
        # F.pad(c8[:,:,1:,:], (0,0,1,0)) would give: row0=0, row1=c8[0], ...
        # No wait. c8[:,:,:-1,:] removes last row. Then pad top with 0.
        # Result: row0=0, rows1..H-1 = c8[rows 0..H-2]
        # So result[r,c] = c8[r-1,c] for r>=1, 0 for r=0. YES that's top neighbor.
        
        top_n = F.pad(c8[:, :, :-1, :], (0, 0, 1, 0))   # c8[r-1,c]
        bot_n = F.pad(c8[:, :, 1:, :], (0, 0, 0, 1))     # c8[r+1,c]
        left_n = F.pad(c8[:, :, :, :-1], (1, 0, 0, 0))   # c8[r,c-1]
        right_n = F.pad(c8[:, :, :, 1:], (0, 1, 0, 0))   # c8[r,c+1]
        
        # Detect L corners (any of 4 types)
        corner_tl = top_n * left_n    # top and left both 8
        corner_tr = top_n * right_n   # top and right both 8
        corner_bl = bot_n * left_n    # bottom and left both 8
        corner_br = bot_n * right_n   # bottom and right both 8
        
        any_corner = torch.clamp(corner_tl + corner_tr + corner_bl + corner_br, 0.0, 1.0)
        
        # Only apply to pixels that are currently 0 (background)
        to_fill = any_corner * c0
        
        # Build output: replace filled pixels from c0 to c1
        out = x.clone()
        out[:, 0:1, :, :] = out[:, 0:1, :, :] * (1.0 - to_fill)
        out[:, 1:2, :, :] = out[:, 1:2, :, :] + to_fill
        
        return out * valid_mask

def export_onnx():
    net = TaskNet()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task081.onnx"
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
    
    # Fix external data
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
