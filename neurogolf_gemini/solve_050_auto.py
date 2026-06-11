import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        is_8 = x[:, 8:9, :, :]
        is_0 = x[:, 0:1, :, :]
        
        left_8 = torch.cumsum(is_8, dim=3)
        right_8 = torch.flip(torch.cumsum(torch.flip(is_8, dims=[3]), dim=3), dims=[3])
        row_match = (left_8 > 0.5) & (right_8 > 0.5)
        
        up_8 = torch.cumsum(is_8, dim=2)
        down_8 = torch.flip(torch.cumsum(torch.flip(is_8, dims=[2]), dim=2), dims=[2])
        col_match = (up_8 > 0.5) & (down_8 > 0.5)
        
        mask_3 = (row_match | col_match) & (is_0 > 0.5)
        mask_3_f = mask_3.float()
        
        out_list = []
        for i in range(10):
            if i == 3:
                out_list.append(x[:, 3:4, :, :] + mask_3_f)
            elif i == 0:
                out_list.append(x[:, 0:1, :, :] - mask_3_f)
            else:
                out_list.append(x[:, i:i+1, :, :])
        return torch.cat(out_list, dim=1)

