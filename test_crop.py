import torch
import numpy as np

content = torch.arange(1, 26).view(1, 1, 5, 5).float()
print("Original:")
print(content[0, 0])

# keep rows 1, 3 and cols 0, 4 (0-indexed)
keep_row = torch.tensor([0, 1, 0, 1, 0]).view(1, 1, 5, 1).float()
keep_col = torch.tensor([1, 0, 0, 0, 1]).view(1, 1, 1, 5).float()

krw = keep_row.view(1, 1, 1, 5)
pr0 = torch.cumsum(krw, dim=3)
pr = pr0 - 1.0
iota_r = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
ErB = (iota_r == pr)
Sr = ErB.float() * krw

pc0 = torch.cumsum(keep_col, dim=3)
pcw = pc0 - 1.0
pc = pcw.view(1, 1, 5, 1)
kc = keep_col.view(1, 1, 5, 1)
iota_c = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5)
EcB = (pc == iota_c)
T = EcB.float() * kc

out = torch.matmul(torch.matmul(Sr, content), T)
print("\nSr:")
print(Sr[0, 0])
print("\nT:")
print(T[0, 0])
print("\nOutput:")
print(out[0, 0])
