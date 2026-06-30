import json
import torch

fname = '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task025.json'
with open(fname) as f:
    data = json.load(f)

ex = data['test'][0]
inp = torch.tensor(ex['input']).unsqueeze(0).unsqueeze(0)
H, W = inp.shape[2], inp.shape[3]

x = torch.zeros(1, 10, H, W)
x.scatter_(1, inp, 1)
x_colors = x[:, 1:, :, :]

row_counts = x_colors.sum(dim=3, keepdim=True)
col_counts = x_colors.sum(dim=2, keepdim=True)

is_hline_mask = (row_counts > W / 2.0).float()
is_vline_mask = (col_counts > H / 2.0).float()

hline = is_hline_mask.expand(-1, -1, -1, W)
vline = is_vline_mask.expand(-1, -1, H, -1)

line_M = torch.clamp(hline + vline, 0.0, 1.0) * x_colors
stray_M = x_colors - line_M

has_hline = torch.max(is_hline_mask, dim=2, keepdim=True)[0]
has_vline = torch.max(is_vline_mask, dim=3, keepdim=True)[0]

y_indices = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
L_y_idx = torch.argmax(is_hline_mask, dim=2, keepdim=True).float()

is_above = (y_indices < L_y_idx).float() * has_hline
is_below = (y_indices > L_y_idx).float() * has_hline

stray_above_cols = torch.max(stray_M * is_above, dim=2, keepdim=True)[0]
stray_below_cols = torch.max(stray_M * is_below, dim=2, keepdim=True)[0]

out_above = (y_indices == L_y_idx - 1).float() * stray_above_cols
out_below = (y_indices == L_y_idx + 1).float() * stray_below_cols

x_indices = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
L_x_idx = torch.argmax(is_vline_mask, dim=3, keepdim=True).float()

is_left = (x_indices < L_x_idx).float() * has_vline
is_right = (x_indices > L_x_idx).float() * has_vline

stray_left_rows = torch.max(stray_M * is_left, dim=3, keepdim=True)[0]
stray_right_rows = torch.max(stray_M * is_right, dim=3, keepdim=True)[0]

out_left = (x_indices == L_x_idx - 1).float() * stray_left_rows
out_right = (x_indices == L_x_idx + 1).float() * stray_right_rows

new_colors = torch.clamp(line_M + out_above + out_below + out_left + out_right, 0.0, 1.0)

out = torch.zeros_like(x)
out[:, 1:, :, :] = new_colors
max_color = torch.max(new_colors, dim=1, keepdim=True)[0]
out[:, 0:1, :, :] = 1.0 - torch.clamp(max_color, 0.0, 1.0)

pred = torch.argmax(out, dim=1).squeeze().numpy()
gt = torch.tensor(ex['output']).numpy()

if not (pred == gt).all():
    print('Failed!')
    for r in range(H):
        if not (pred[r] == gt[r]).all():
            print(f'Row {r} mismatch!')
            print(f'  Pred: {pred[r]}')
            print(f'  GT  : {gt[r]}')
else:
    print('Success!')
