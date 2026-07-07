import torch
import torch.nn.functional as F

H, W = 5, 5
s = 2

m_A = torch.zeros(1, 1, H, W)
m_B = torch.zeros(1, 1, H, W)

# Target: A at (0..1, 0..1), B at (0..1, 4..5) Wait, W=5 so (0..1, 3..4). Distance 3.
m_A[0, 0, 0:2, 0:2] = 1
m_B[0, 0, 0:2, 3:5] = 1

# Template: A at (0,0), B at (0,1). Distance 1. Wait, scaled to 2, it would be distance 2!
# Let's make template A at (0,0), B at (0,2). Wait, then scaled it's distance 4!
# Let's adjust template to A at (0,0), B at (0,1). Scaled to 2, distance is 2.
# So target distance MUST BE 2!
m_B = torch.zeros(1, 1, H, W)
m_B[0, 0, 0:2, 2:4] = 1 # A is 0..1, B is 2..3. Distance 2!

T_A = torch.zeros(1, 1, H, W)
T_B = torch.zeros(1, 1, H, W)
T_path = torch.zeros(1, 1, H, W)

T_A[0, 0, 0, 0] = 1
T_B[0, 0, 0, 1] = 1
T_path[0, 0, 1, 0] = 1

T_A_s = F.interpolate(T_A, scale_factor=s, mode='nearest')
T_B_s = F.interpolate(T_B, scale_factor=s, mode='nearest')
T_path_s = F.interpolate(T_path, scale_factor=s, mode='nearest')

sum_A = T_A_s.sum()
sum_B = T_B_s.sum()

pad_h = s * H
pad_w = s * W
m_A_padded = F.pad(m_A, (pad_w, pad_w, pad_h, pad_h))
m_B_padded = F.pad(m_B, (pad_w, pad_w, pad_h, pad_h))

corr_A = F.conv2d(m_A_padded, T_A_s)
corr_B = F.conv2d(m_B_padded, T_B_s)

valid_A = (corr_A >= sum_A - 0.5)
valid_B = (corr_B >= sum_B - 0.5)
valid_shift = valid_A & valid_B

print("valid_shift nonzeros:", torch.nonzero(valid_shift))

kernel = torch.flip(valid_shift.float(), [2, 3])
shifted_path = F.conv2d(T_path_s, kernel)

# Wait, the output size of shifted_path:
# T_path_s is sH x sW. kernel is (H + 2sH - sH + 1) = H + sH + 1.
# kernel > T_path_s!
# F.conv2d requires input >= kernel!
# So we MUST pad T_path_s!
# Let's pad T_path_s to match the original padding.

