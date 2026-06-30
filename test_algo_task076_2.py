import torch
import torch.nn.functional as F

def transform(x, k):
    if k >= 4:
        x = torch.flip(x, [3])
        k = k - 4
    if k == 1:
        x = torch.rot90(x, 1, [2, 3])
    elif k == 2:
        x = torch.rot90(x, 2, [2, 3])
    elif k == 3:
        x = torch.rot90(x, 3, [2, 3])
    return x

def get_shifted(src, ref, target):
    # We want to shift src by the same offset that shifts ref to target.
    # We can do this by computing cross-correlation between target and ref to find the shift.
    # target: [B, 1, H, W], ref: [B, 1, H, W]
    # corr: [B, 1, 2H-1, 2W-1]
    B, _, H, W = target.shape
    # To find the shift, we convolve target with ref.
    # F.conv2d takes weight of shape [out_c, in_c, kH, kW].
    # But weight cannot have batch dimension in F.conv2d.
    # We can use grouped convolution where groups=B!
    target_g = target.view(1, B, H, W)
    ref_g = ref.view(B, 1, H, W)
    corr = F.conv2d(target_g, ref_g, padding=(H-1, W-1), groups=B)
    # corr is [1, B, 2H-1, 2W-1]
    # A value of 1 in corr at (y, x) means target and ref align when ref is shifted by (y - (H-1), x - (W-1)).
    # We want to apply this same shift to src!
    # So we convolve src with the flipped corr!
    # Wait, if we convolve src with corr, what does it mean?
    
    # Actually, shifting src by (dy, dx) is:
    # out(i, j) = src(i - dy, j - dx)
    # If shift mask has 1 at (H-1 + dy, W-1 + dx).
    # Then we can just convolve shift_mask with src!
    # We want out(i, j) = sum_{dy, dx} shift_mask(dy, dx) * src(i - dy, j - dx)
    # This is exactly convolution of shift_mask with src!
    
    # Wait, let's test it with a simple example.
    pass

H, W = 5, 5
target = torch.zeros(1, 1, H, W)
ref = torch.zeros(1, 1, H, W)
src = torch.zeros(1, 1, H, W)

target[0, 0, 3, 3] = 1 # target is at (3,3)
ref[0, 0, 1, 1] = 1    # ref is at (1,1)
src[0, 0, 1, 2] = 1    # src is at (1,2)

# Shift should be (2, 2)
# Expected shifted_src is at (1+2, 2+2) = (3, 4)

target_g = target.view(1, 1, H, W)
ref_g = ref.view(1, 1, H, W)
corr = F.conv2d(target_g, ref_g, padding=(H-1, W-1))

print("Corr max pos:", torch.nonzero(corr))
# Output will be at (0, 0, 4+2, 4+2) = (0, 0, 6, 6) because H-1=4.
# Wait, H=5. H-1=4.
# (3, 3) - (1, 1) = (2, 2). Offset + 4 = 6. So (6, 6). Correct!

corr_flip = torch.flip(corr, [2, 3])
src_g = src.view(1, 1, H, W)
shifted_src = F.conv2d(corr, src_g, padding=0)
print("Shifted src pos:", torch.nonzero(shifted_src))

