import torch
import torch.nn as nn
import os

class Model(nn.Module):
    def forward(self, x):
        # x is [B, 10, 30, 30]
        is_valid = x.sum(dim=1) > 0.5 # [B, 30, 30]
        is_8 = x[:, 8, :, :] > 0.5 # [B, 30, 30]
        
        row_is_8 = is_8.float().amax(dim=-1, keepdim=True) # [B, 30, 1]
        col_is_8 = is_8.float().amax(dim=-2, keepdim=True) # [B, 1, 30]
        
        row_cumsum = torch.cumsum(row_is_8, dim=-2)
        col_cumsum = torch.cumsum(col_is_8, dim=-1)
        
        row_is_8_bool = row_is_8 > 0.5
        col_is_8_bool = col_is_8 > 0.5
        
        top_mask = (row_cumsum < 0.5)
        mid_mask = (row_cumsum > 0.5) & (row_cumsum < 1.5) & ~row_is_8_bool
        bot_mask = (row_cumsum > 1.5) & ~row_is_8_bool
        
        left_mask = (col_cumsum < 0.5)
        center_mask = (col_cumsum > 0.5) & (col_cumsum < 1.5) & ~col_is_8_bool
        right_mask = (col_cumsum > 1.5) & ~col_is_8_bool
        
        top_center = top_mask & center_mask & is_valid
        bot_center = bot_mask & center_mask & is_valid
        mid_left = mid_mask & left_mask & is_valid
        mid_center = mid_mask & center_mask & is_valid
        mid_right = mid_mask & right_mask & is_valid
        
        top_left = top_mask & left_mask & is_valid
        top_right = top_mask & right_mask & is_valid
        bot_left = bot_mask & left_mask & is_valid
        bot_right = bot_mask & right_mask & is_valid
        
        grid_mask = is_8 & is_valid
        
        out0 = (top_left | top_right | bot_left | bot_right).float()
        out1 = bot_center.float()
        out2 = top_center.float()
        out3 = mid_right.float()
        out4 = mid_left.float()
        out5 = torch.zeros_like(out0)
        out6 = mid_center.float()
        out7 = torch.zeros_like(out0)
        out8 = grid_mask.float()
        out9 = torch.zeros_like(out0)
        
        out = torch.stack([out0, out1, out2, out3, out4, out5, out6, out7, out8, out9], dim=1)
        
        return out

def build_model():
    model = Model()
    model.eval()
    return model

if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.zeros((1, 10, 30, 30), dtype=torch.float32)
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task055.onnx"
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    )
    print(f"Exported to {onnx_path}")
