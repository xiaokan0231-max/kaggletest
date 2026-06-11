import torch
import torch.nn as nn
import torch.nn.functional as F

class Task035Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is (B, C, H, W)
        
        # R: moves right ->
        R = x.clone()
        R = torch.where(x == 8, torch.tensor(0, dtype=x.dtype, device=x.device), R)
        for _ in range(30):
            shifted = F.pad(R[..., :-1], (1, 0), value=0)
            R = torch.where((x == 0) & (R == 0), shifted, R)
        color_L = F.pad(R[..., :-1], (1, 0), value=0)

        # L: moves left <-
        L = x.clone()
        L = torch.where(x == 8, torch.tensor(0, dtype=x.dtype, device=x.device), L)
        for _ in range(30):
            shifted = F.pad(L[..., 1:], (0, 1), value=0)
            L = torch.where((x == 0) & (L == 0), shifted, L)
        color_R = F.pad(L[..., 1:], (0, 1), value=0)

        # D: moves down v
        D = x.clone()
        D = torch.where(x == 8, torch.tensor(0, dtype=x.dtype, device=x.device), D)
        for _ in range(30):
            shifted = F.pad(D[..., :-1, :], (0, 0, 1, 0), value=0)
            D = torch.where((x == 0) & (D == 0), shifted, D)
        color_T = F.pad(D[..., :-1, :], (0, 0, 1, 0), value=0)

        # U: moves up ^
        U = x.clone()
        U = torch.where(x == 8, torch.tensor(0, dtype=x.dtype, device=x.device), U)
        for _ in range(30):
            shifted = F.pad(U[..., 1:, :], (0, 0, 0, 1), value=0)
            U = torch.where((x == 0) & (U == 0), shifted, U)
        color_B = F.pad(U[..., 1:, :], (0, 0, 0, 1), value=0)

        # Identify edges
        x_left = F.pad(x[..., :-1], (1, 0), value=0)
        left_edge = (x == 8) & (x_left != 8)

        x_right = F.pad(x[..., 1:], (0, 1), value=0)
        right_edge = (x == 8) & (x_right != 8)

        x_top = F.pad(x[..., :-1, :], (0, 0, 1, 0), value=0)
        top_edge = (x == 8) & (x_top != 8)

        x_bottom = F.pad(x[..., 1:, :], (0, 0, 0, 1), value=0)
        bottom_edge = (x == 8) & (x_bottom != 8)

        new_x = x.clone()

        update_L = left_edge & (color_L != 0)
        new_x = torch.where(update_L, color_L, new_x)

        update_R = right_edge & (color_R != 0)
        new_x = torch.where(update_R, color_R, new_x)

        update_T = top_edge & (color_T != 0)
        new_x = torch.where(update_T, color_T, new_x)

        update_B = bottom_edge & (color_B != 0)
        new_x = torch.where(update_B, color_B, new_x)

        return new_x

if __name__ == "__main__":
    model = Task035Network()
    dummy_input = torch.zeros((1, 1, 10, 10), dtype=torch.int32)
    torch.onnx.export(
        model, 
        dummy_input, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task035.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=14,
        dynamic_axes={"input": {2: "height", 3: "width"}, "output": {2: "height", 3: "width"}}
    )
    print("ONNX model exported successfully.")
