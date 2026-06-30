import torch
import torch.nn.functional as F

T_A = torch.zeros(1, 1, 5, 5)
T_A[0, 0, 1, 1] = 1

T_A_s = F.interpolate(T_A, scale_factor=3, mode='nearest')
print(T_A_s.shape)
print(torch.nonzero(T_A_s))
