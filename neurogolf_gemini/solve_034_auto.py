import torch
import torch.nn as nn
import torch.nn.functional as F

class Task034Network(nn.Module):
    def __init__(self):
        super().__init__()
        
    def shift_mask(self, mask, dr, dc):
        pad_t = max(0, dr)
        pad_b = max(0, -dr)
        pad_l = max(0, dc)
        pad_r = max(0, -dc)
        padded = F.pad(mask, (pad_l, pad_r, pad_t, pad_b))
        H, W = mask.shape[2], mask.shape[3]
        return padded[:, :, pad_b : pad_b + H, pad_r : pad_r + W]

    def build_beam(self, M_box, dr, dc):
        beam = M_box
        for i in range(5):
            shift_amount = 1 << i
            beam = beam + self.shift_mask(beam, dr * shift_amount, dc * shift_amount)
            beam = (beam > 0.5).float()
        return beam

    def dilate_to_2x2(self, source):
        s00 = source
        s01 = self.shift_mask(source, 0, 1)
        s10 = self.shift_mask(source, 1, 0)
        s11 = self.shift_mask(source, 1, 1)
        res = s00 + s01 + s10 + s11
        return (res > 0.5).float()

    def forward(self, x):
        # x is (B, 1, H, W)
        M = (x > 0).float()
        
        # 2x2 conv to find full 2x2 boxes
        kernel = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=x.device)
        conv = F.conv2d(M, kernel, padding=0)
        
        P_TL = (conv == 4.0).float()
        P_TL = F.pad(P_TL, (0, 1, 0, 1))
        
        M2 = (x == 2).float()
        
        # Determine source pixels for each direction
        Source_TL = M2 * P_TL
        
        M2_left = self.shift_mask(M2, 0, -1)
        Source_TR = M2_left * P_TL
        
        M2_up = self.shift_mask(M2, -1, 0)
        Source_BL = M2_up * P_TL
        
        M2_up_left = self.shift_mask(M2, -1, -1)
        Source_BR = M2_up_left * P_TL
        
        # Reconstruct the moving boxes
        Box_TL = self.dilate_to_2x2(Source_TL)
        Box_TR = self.dilate_to_2x2(Source_TR)
        Box_BL = self.dilate_to_2x2(Source_BL)
        Box_BR = self.dilate_to_2x2(Source_BR)
        
        # Build beams
        beam_TL = self.build_beam(Box_TL, -1, -1)
        beam_TR = self.build_beam(Box_TR, -1, 1)
        beam_BL = self.build_beam(Box_BL, 1, -1)
        beam_BR = self.build_beam(Box_BR, 1, 1)
        
        All_Boxes = self.dilate_to_2x2(P_TL)
        
        final_mask = All_Boxes + beam_TL + beam_TR + beam_BL + beam_BR
        final_mask = (final_mask > 0.5).float()
        
        # Extract base color (max value that is not 2)
        # Using view to safely get global max over spatial dims without using unsupported dynamic repeats
        B = x.shape[0]
        x_not_2 = x * (x != 2).float()
        C = x_not_2.view(B, 1, -1).max(dim=2)[0].view(B, 1, 1, 1)
        
        out = final_mask * C
        return out.to(torch.int32)

if __name__ == "__main__":
    import os
    model = Task034Network()
    dummy_input = torch.zeros((1, 1, 30, 30), dtype=torch.float32)
    torch.onnx.export(
        model, 
        dummy_input, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task034.onnx",
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}}
    )
    print("Exported task034.onnx")
