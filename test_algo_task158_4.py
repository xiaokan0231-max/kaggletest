import torch
import torch.nn.functional as F

H, W = 5, 5
s = 2

# Original Grid: 5x5
m_A = torch.zeros(1, 1, H, W)
m_B = torch.zeros(1, 1, H, W)

# Target: A at (0, 0), B at (0, 2). Size 1x1.
# So distance is 2.
m_A[0, 0, 0, 0] = 1
m_B[0, 0, 0, 2] = 1

# Template: A at (0,0), B at (0,1). Size 1x1.
T_A = torch.zeros(1, 1, H, W)
T_B = torch.zeros(1, 1, H, W)
T_path = torch.zeros(1, 1, H, W)

T_A[0, 0, 0, 0] = 1
T_B[0, 0, 0, 1] = 1
T_path[0, 0, 1, 0] = 1  # path at (1,0)

# Scale by s=2
T_A_s = F.interpolate(T_A, scale_factor=s, mode='nearest')
T_B_s = F.interpolate(T_B, scale_factor=s, mode='nearest')
T_path_s = F.interpolate(T_path, scale_factor=s, mode='nearest')

# T_A_s has A at (0..1, 0..1). T_B_s has B at (0..1, 2..3).
# Wait, in the target, m_A and m_B are 1x1!
# If T_A_s is 2x2, it will NEVER overlap with a 1x1 m_A! sum_A will be 4, but max corr_A will be 1!
# Ah! Target must ALSO be scaled? No, the target in ARC IS scaled!
pass
