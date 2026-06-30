import torch
mask_r = torch.tensor([0, 0, 1, 0, 1, 0, 0]).view(1, 1, 7, 1)
prefix_sum = torch.cumsum(mask_r, dim=2)
suffix_sum = torch.flip(torch.cumsum(torch.flip(mask_r, [2]), dim=2), [2])
keep_row = (prefix_sum > 0) & (suffix_sum > 0)
print("Mask R:")
print(mask_r.view(-1))
print("Keep Row:")
print(keep_row.view(-1))
