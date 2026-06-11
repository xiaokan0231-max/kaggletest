import torch
import torch.nn as nn
import onnx

class Task028Net(nn.Module):
    def forward(self, x):
        # x is (1, 10, H, W)
        # We know H=10, W=10
        # Wait, the pad to 30x30 will pass 30x30 to us!
        # But valid_mask will be 10x10
        
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        
        out = torch.zeros_like(x)
        
        # We need to find the colors in top half and bottom half
        # Since it's padded to 30x30, the top half of the VALID region is rows 0..4
        # Bottom half is rows 5..9
        
        # To make it robust for any shape (e.g. H=10), we can just slice.
        # Let's assume H=10.
        top_half = x[:, :, 0:5, :]
        bottom_half = x[:, :, 5:10, :]
        
        # Find which color is present in top half
        # c_top has shape (1, 10, 1, 1)
        c_top = torch.max(torch.max(top_half, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        c_top[:, 0:1, :, :] = 0
        
        c_bot = torch.max(torch.max(bottom_half, dim=2, keepdim=True)[0], dim=3, keepdim=True)[0]
        c_bot[:, 0:1, :, :] = 0
        
        # Draw vertical lines in col 0 and 9
        # top half:
        out[:, :, 0:5, 0:1] = c_top
        out[:, :, 0:5, 9:10] = c_top
        # bottom half:
        out[:, :, 5:10, 0:1] = c_bot
        out[:, :, 5:10, 9:10] = c_bot
        
        # Draw horizontal line at row 0 (top half)
        out[:, :, 0:1, 0:10] = c_top
        
        # Draw horizontal line at row 9 (bottom half)
        out[:, :, 9:10, 0:10] = c_bot
        
        # Draw horizontal line at the row where the pixel was
        # For top half, row has pixel if max over W is 1
        # top_row_mask: (1, 10, 5, 1)
        top_row_mask = torch.max(top_half, dim=3, keepdim=True)[0]
        out[:, :, 0:5, 0:10] = torch.max(out[:, :, 0:5, 0:10], top_row_mask)
        
        # bottom_row_mask: (1, 10, 5, 1)
        bot_row_mask = torch.max(bottom_half, dim=3, keepdim=True)[0]
        out[:, :, 5:10, 0:10] = torch.max(out[:, :, 5:10, 0:10], bot_row_mask)
        
        # Background
        # Zero out channel 0 first because we might have accidentally copied it
        out[:, 0:1, :, :] = 0
        max_color = torch.max(out, dim=1, keepdim=True)[0]
        bg = 1 - torch.clamp(max_color, 0, 1)
        out[:, 0:1, :, :] = bg
        
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task028Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task028.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
