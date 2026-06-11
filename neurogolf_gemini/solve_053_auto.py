import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # W to shift down by 1 row
        self.W = nn.Parameter(torch.zeros(10, 10, 3, 3), requires_grad=False)
        for i in range(10):
            self.W[i, i, 0, 1] = 1.0
            
        # W2 to broadcast 1 channel to 10 channels (1 at ch0, 0 elsewhere)
        self.W2 = nn.Parameter(torch.zeros(10, 1, 1, 1), requires_grad=False)
        self.W2[0, 0, 0, 0] = 1.0

    def forward(self, x):
        shifted = F.conv2d(x, self.W, padding=1)
        
        grid_mask = torch.sum(x, dim=1, keepdim=True)
        masked_shifted = shifted * grid_mask
        
        empty_cells = grid_mask - torch.sum(masked_shifted, dim=1, keepdim=True)
        
        zero_fill = F.conv2d(empty_cells, self.W2, padding=0)
        
        out = masked_shifted + zero_fill
        return out
