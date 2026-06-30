import torch
import torch.nn as nn
import json, numpy as np

def crop_compact(x, pr, sr, pc, sc):
    mask_r = (pr > 0.5) & (sr > 0.5)
    mask_c = (pc > 0.5) & (sc > 0.5)
    Sr = (mask_r.float() == pr).float()
    Sc = (mask_c.float() == pc).float()
    return torch.matmul(torch.matmul(Sr.transpose(2, 3), x), Sc)

class AssemblyNet(nn.Module):
    def __init__(self, box_color):
        super().__init__()
        self.box_color = box_color

    def dynamic_roll(self, x, shift_y, shift_x):
        iota_r = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 30, 1)
        target_idx_r = torch.remainder(iota_r - shift_y.float(), 30.0)
        iota_c = torch.arange(30, device=x.device, dtype=torch.float32).view(1, 1, 1, 30)
        P_r = (iota_c == target_idx_r).float()
        
        target_idx_c = torch.remainder(iota_c - shift_x.float(), 30.0)
        P_c = (iota_r == target_idx_c).float()
        
        return torch.matmul(torch.matmul(P_r, x), P_c)

    def forward(self, x):
        mask_box = x[:, self.box_color:self.box_color+1, :, :]
        
        sums = torch.sum(x, dim=(2, 3))
        sums[:, 0] = -1
        sums[:, self.box_color] = -1
        obj_color_idx = torch.argmax(sums, dim=1)
        obj_color_onehot = torch.nn.functional.one_hot(obj_color_idx, num_classes=10).float()
        mask_obj = torch.sum(x * obj_color_onehot.view(-1, 10, 1, 1), dim=1, keepdim=True)
        
        # Bbox for obj
        obj_r = torch.max(mask_obj, dim=3, keepdim=True)[0]
        obj_pr = torch.cumsum(obj_r, dim=2)
        obj_sr = torch.flip(torch.cumsum(torch.flip(obj_r, [2]), dim=2), [2])
        obj_c = torch.max(mask_obj, dim=2, keepdim=True)[0]
        obj_pc = torch.cumsum(obj_c, dim=3)
        obj_sc = torch.flip(torch.cumsum(torch.flip(obj_c, [3]), dim=3), [3])
        
        obj_h = torch.sum(((obj_pr > 0.5) & (obj_sr > 0.5)).float(), dim=2).view(-1).long()
        obj_w = torch.sum(((obj_pc > 0.5) & (obj_sc > 0.5)).float(), dim=3).view(-1).long()
        obj_h = torch.clamp(obj_h, min=1)
        obj_w = torch.clamp(obj_w, min=1)
        
        # Bbox for box
        box_r = torch.max(mask_box, dim=3, keepdim=True)[0]
        box_pr = torch.cumsum(box_r, dim=2)
        box_sr = torch.flip(torch.cumsum(torch.flip(box_r, [2]), dim=2), [2])
        box_c = torch.max(mask_box, dim=2, keepdim=True)[0]
        box_pc = torch.cumsum(box_c, dim=3)
        box_sc = torch.flip(torch.cumsum(torch.flip(box_c, [3]), dim=3), [3])
        
        box_h = torch.sum(((box_pr > 0.5) & (box_sr > 0.5)).float(), dim=2).view(-1).long()
        box_w = torch.sum(((box_pc > 0.5) & (box_sc > 0.5)).float(), dim=3).view(-1).long()
        
        scale_factor = torch.div(box_h - 2, obj_h, rounding_mode='floor')
        
        obj_compacted = crop_compact(mask_obj, obj_pr, obj_sr, obj_pc, obj_sc)
        box_compacted = crop_compact(mask_box, box_pr, box_sr, box_pc, box_sc)
        
        obj_scaled = torch.repeat_interleave(obj_compacted, scale_factor.item(), dim=2)
        obj_scaled = torch.repeat_interleave(obj_scaled, scale_factor.item(), dim=3)
        
        # Wait, if obj_scaled is larger than 30, it will crash. Let's pad it to 30!
        # Actually, repeat_interleave increases the shape of the tensor!
        # This breaks static shape constraint of ONNX if we use dynamic repeat_interleave!
        # Wait! In test_dynamic_scale.py, I used repeat_interleave with a dynamic factor. Did it keep shape 30?
        # NO! It changed shape! In ONNX, the output shape would be dynamic!
        # We need to do scaling WITHOUT changing the tensor shape (keeping it 30x30).
        return obj_scaled

net = AssemblyNet(box_color=2)

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    d = json.load(f)
ex = d['train'][0]
inp = np.array(ex['input'])
x = torch.zeros(1, 10, 30, 30)
for r in range(inp.shape[0]):
    for c in range(inp.shape[1]):
        x[0, inp[r,c], r, c] = 1.0

out = net(x)
print(out.shape)
