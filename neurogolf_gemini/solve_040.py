import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task040Net(nn.Module):
    def forward(self, x):
        # x is (B, 10, H, W) one-hot
        
        # Anchors: all colors except 0 and 3
        anchors = x.clone()
        anchors[:, 0, :, :] = 0
        anchors[:, 3, :, :] = 0
        
        curr_mask = torch.max(anchors, dim=1, keepdim=True)[0] # (B, 1, H, W)
        color_field = anchors.clone()
        
        # Iteratively expand
        for _ in range(30):
            next_mask = F.max_pool2d(curr_mask, kernel_size=3, stride=1, padding=1)
            new_pixels = next_mask - curr_mask
            
            # Pool color_field
            next_color_field = F.max_pool2d(color_field, kernel_size=3, stride=1, padding=1)
            
            # Update color_field at new_pixels
            color_field = color_field + next_color_field * new_pixels
            curr_mask = next_mask
            
        # We only want to replace 3s!
        is_3 = x[:, 3:4, :, :]
        
        # For pixels that are 3, we use color_field.
        # But color_field might have multiple 1s in channel dim if there was a tie.
        # To make it a valid one-hot, we can just use argmax, then one_hot again.
        # However, to keep it 0-param and simple:
        # We can just pick the channel with the max value.
        # But color_field is already composed of 1s and 0s. 
        # Let's just create a new one-hot from color_field.
        # Since argmax is not supported by some ONNX operators without params?
        # Actually torch.argmax is supported.
        # Let's just do:
        c_max = torch.max(color_field, dim=1, keepdim=True)[0]
        # Only keep the highest channel index in case of ties?
        # A simple way to break ties:
        weights = torch.arange(10, dtype=torch.float32, device=x.device).view(1, 10, 1, 1)
        weighted_field = color_field * weights
        best_c = torch.argmax(weighted_field, dim=1, keepdim=True)
        
        # Create one-hot from best_c
        best_color_onehot = []
        for c in range(10):
            best_color_onehot.append((best_c == c).float())
        best_color_onehot = torch.cat(best_color_onehot, dim=1)
        
        out = x * (1.0 - is_3) + best_color_onehot * is_3
        
        # Apply valid mask
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out = out * valid_mask
        
        return out

def export_onnx():
    net = Task040Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task040.onnx"
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
