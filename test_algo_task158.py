import torch
import torch.nn.functional as F

def get_bounding_box(mask):
    # mask: [1, 1, H, W]
    row_sum = torch.sum(mask, dim=(0, 1, 3))
    col_sum = torch.sum(mask, dim=(0, 1, 2))
    row_indices = torch.nonzero(row_sum)
    col_indices = torch.nonzero(col_sum)
    if len(row_indices) == 0:
        return 0, 1, 0, 1
    r_min = row_indices.min().item()
    r_max = row_indices.max().item() + 1
    c_min = col_indices.min().item()
    c_max = col_indices.max().item() + 1
    return r_min, r_max, c_min, c_max

# let's test if we can do bounding box dynamically in ONNX?
# No, dynamic cropping is very hard in ONNX without dynamic axes.
# But wait! We don't NEED to crop dynamically!
# If we just shift the template to the TOP-LEFT of the 30x30 grid!
# How to shift to top-left?
# We can find r_min, c_min using torch.argmax on cumsum!
pass
