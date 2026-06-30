import torch
import torch.nn.functional as F

H, W = 5, 5
target = torch.zeros(1, 1, H, W)
ref = torch.zeros(1, 1, H, W)
src = torch.zeros(1, 1, H, W)

target[0, 0, 3, 3] = 1 # target is at (3,3)
target[0, 0, 0, 1] = 1 # another target at (0,1)
ref[0, 0, 1, 1] = 1    # ref is at (1,1)
src[0, 0, 1, 2] = 1    # src is at (1,2)

target_g = target.view(1, 1, H, W)
ref_g = ref.view(1, 1, H, W)
corr = F.conv2d(target_g, ref_g, padding=(H-1, W-1))

kernel = torch.flip(corr, [2, 3])
src_padded = F.pad(src, (W-1, W-1, H-1, H-1))
shifted_src = F.conv2d(src_padded, kernel)

print("Shifted src pos:", torch.nonzero(shifted_src))

