import torch
import torch.nn.functional as F

H, W = 5, 5
target = torch.zeros(1, 1, H, W)
ref = torch.zeros(1, 1, H, W)
src = torch.zeros(1, 1, H, W)

target[0, 0, 3, 3] = 1 # target is at (3,3)
ref[0, 0, 1, 1] = 1    # ref is at (1,1)
src[0, 0, 1, 2] = 1    # src is at (1,2)

target_g = target.view(1, 1, H, W)
ref_g = ref.view(1, 1, H, W)
# find shift: target = ref + (dy, dx)
# corr[dy + H-1, dx + W-1] = 1
corr = F.conv2d(target_g, ref_g, padding=(H-1, W-1))

# Now we want to shift src by (dy, dx).
# shifted_src = F.conv2d(src, kernel, padding=(H-1, W-1))
# where kernel is corr flipped?
kernel = torch.flip(corr, [2, 3])
# kernel size is 2H-1. src size is H.
# wait, conv2d requires input >= kernel.
# so we pad src!
src_padded = F.pad(src, (W-1, W-1, H-1, H-1))
shifted_src = F.conv2d(src_padded, kernel)
# output size: (H + 2(H-1)) - (2H-1) + 1 = 3H - 2 - 2H + 1 + 1 = H
print("Shifted src pos:", torch.nonzero(shifted_src))

