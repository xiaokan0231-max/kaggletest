import torch
import torch.nn as nn
import onnx

class Task032Net(nn.Module):
    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        # Get class indices (0-9). Padded regions will be 0.
        grid = torch.argmax(x, dim=1, keepdim=True).float()
        
        # Set padded regions to 10.0 so they sort to the bottom
        grid_sort = grid + (1.0 - valid_mask) * 10.0
        
        # Sort ascending: 0s go to top, non-zeros go to bottom of valid region, 10s go to padding region
        sorted_grid, _ = torch.sort(grid_sort, dim=2, descending=False)
        
        # Mask out the padding region and convert back to int64
        final_grid = (sorted_grid * valid_mask).long()
        
        # Convert to one-hot
        out = torch.zeros_like(x)
        out.scatter_(1, final_grid, 1.0)
        
        # For padding, scatter will put 1.0 at channel 0. But valid_mask is 0 there.
        # We should mask out the output to be exactly 0 in padding regions to match neurogolf format.
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task032Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task032.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}},
        dynamo=False
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
