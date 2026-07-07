import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_ray(start_mask, obs_mask, dim, flip):
    if flip:
        start_mask = torch.flip(start_mask, [dim])
        obs_mask = torch.flip(obs_mask, [dim])
        
    cs_start = torch.cumsum(start_mask, dim=dim)
    ray_fwd = (cs_start > 0.5).float()
    obs_after = obs_mask * ray_fwd
    cs_obs = torch.cumsum(obs_after, dim=dim)
    ray = ray_fwd * (cs_obs < 0.5).float()
    
    if flip:
        ray = torch.flip(ray, [dim])
    return ray

def get_dist(ray_mask, dim, flip):
    if flip:
        rm = torch.flip(ray_mask, [dim])
        dist = torch.cumsum(rm, dim=dim)
        dist = torch.flip(dist, [dim])
    else:
        dist = torch.cumsum(ray_mask, dim=dim)
    return dist * ray_mask

class TaskNet066(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        m3 = x[:, 3:4, :, :]
        m2 = x[:, 2:3, :, :]
        m8 = x[:, 8:9, :, :]
        
        k_v = torch.ones(1, 1, 2, 1, device=x.device)
        is_vert3 = (torch.max(F.conv2d(m3, k_v).view(x.shape[0], -1), dim=1)[0] > 1.5).float().view(-1, 1, 1, 1)
        is_horiz3 = 1.0 - is_vert3
        
        is_vert2 = (torch.max(F.conv2d(m2, k_v).view(x.shape[0], -1), dim=1)[0] > 1.5).float().view(-1, 1, 1, 1)
        is_horiz2 = 1.0 - is_vert2
        
        R3_D = get_ray(m3, m8, 2, False)
        R3_U = get_ray(m3, m8, 2, True)
        R3_R = get_ray(m3, m8, 3, False)
        R3_L = get_ray(m3, m8, 3, True)
        
        R2_D = get_ray(m2, m8, 2, False)
        R2_U = get_ray(m2, m8, 2, True)
        R2_R = get_ray(m2, m8, 3, False)
        R2_L = get_ray(m2, m8, 3, True)
        
        Ray3_V = torch.clamp(R3_D + R3_U, 0, 1)
        Dist3_V = get_dist(R3_D, 2, False) + get_dist(R3_U, 2, True)
        Corr3_H = torch.clamp(get_ray(Ray3_V, m8, 3, False) + get_ray(Ray3_V, m8, 3, True), 0, 1)
        
        Ray3_H = torch.clamp(R3_R + R3_L, 0, 1)
        Dist3_H = get_dist(R3_R, 3, False) + get_dist(R3_L, 3, True)
        Corr3_V = torch.clamp(get_ray(Ray3_H, m8, 2, False) + get_ray(Ray3_H, m8, 2, True), 0, 1)
        
        Ray3 = is_vert3 * Ray3_V + is_horiz3 * Ray3_H
        Dist3 = is_vert3 * Dist3_V + is_horiz3 * Dist3_H
        Corridor3 = is_vert3 * Corr3_H + is_horiz3 * Corr3_V
        
        Dist3_V_broadcast = torch.max(Dist3_V, dim=3, keepdim=True)[0]
        Dist3_H_broadcast = torch.max(Dist3_H, dim=2, keepdim=True)[0]
        Dist3_broadcast = is_vert3 * Dist3_V_broadcast + is_horiz3 * Dist3_H_broadcast
        
        Ray2_V = torch.clamp(R2_D + R2_U, 0, 1)
        Ray2_H = torch.clamp(R2_R + R2_L, 0, 1)
        Ray2 = is_vert2 * Ray2_V + is_horiz2 * Ray2_H
        
        Intersect = Corridor3 * Ray2
        
        coord_x = torch.arange(x.shape[3], device=x.device).view(1, 1, 1, x.shape[3]).float() * 0.0001
        coord_y = torch.arange(x.shape[2], device=x.device).view(1, 1, x.shape[2], 1).float() * 0.0001
        noise = coord_x + coord_y
        
        Valid_Dist3 = Dist3_broadcast * Intersect + noise * Intersect
        max_d = torch.max(Valid_Dist3.view(x.shape[0], -1), dim=1)[0].view(-1, 1, 1, 1)
        Chosen_Intersect = ((Valid_Dist3 == max_d) & (Intersect > 0.5)).float()
        
        # Perpendicular path from Chosen_Intersect
        Proj_V = torch.clamp(get_ray(Chosen_Intersect, m8, 3, False) + get_ray(Chosen_Intersect, m8, 3, True), 0, 1) * Ray3
        cs_CI_R = (torch.cumsum(Chosen_Intersect, dim=3) > 0.5).float()
        cs_CI_L = (torch.flip(torch.cumsum(torch.flip(Chosen_Intersect, [3]), dim=3), [3]) > 0.5).float()
        cs_proj_R = (torch.cumsum(Proj_V, dim=3) > 0.5).float()
        cs_proj_L = (torch.flip(torch.cumsum(torch.flip(Proj_V, [3]), dim=3), [3]) > 0.5).float()
        Path_Horiz = torch.clamp(cs_proj_R * cs_CI_L + cs_proj_L * cs_CI_R + Proj_V + Chosen_Intersect, 0, 1)
        
        Proj_H = torch.clamp(get_ray(Chosen_Intersect, m8, 2, False) + get_ray(Chosen_Intersect, m8, 2, True), 0, 1) * Ray3
        cs_CI_D = (torch.cumsum(Chosen_Intersect, dim=2) > 0.5).float()
        cs_CI_U = (torch.flip(torch.cumsum(torch.flip(Chosen_Intersect, [2]), dim=2), [2]) > 0.5).float()
        cs_proj_D = (torch.cumsum(Proj_H, dim=2) > 0.5).float()
        cs_proj_U = (torch.flip(torch.cumsum(torch.flip(Proj_H, [2]), dim=2), [2]) > 0.5).float()
        Path_Vert = torch.clamp(cs_proj_D * cs_CI_U + cs_proj_U * cs_CI_D + Proj_H + Chosen_Intersect, 0, 1)
        
        Path_Perp = is_vert3 * Path_Horiz + is_horiz3 * Path_Vert
        Proj = is_vert3 * Proj_V + is_horiz3 * Proj_H
        
        # Trim Ray3
        cs_m3_D = (torch.cumsum(m3, dim=2) > 0.5).float()
        cs_proj_U = (torch.flip(torch.cumsum(torch.flip(Proj_V, [2]), dim=2), [2]) > 0.5).float()
        cs_m3_U = (torch.flip(torch.cumsum(torch.flip(m3, [2]), dim=2), [2]) > 0.5).float()
        cs_proj_D = (torch.cumsum(Proj_V, dim=2) > 0.5).float()
        Path3_V = torch.clamp(cs_m3_D * cs_proj_U + cs_m3_U * cs_proj_D + m3 + Proj_V, 0, 1) * Ray3
        
        cs_m3_R = (torch.cumsum(m3, dim=3) > 0.5).float()
        cs_proj_L = (torch.flip(torch.cumsum(torch.flip(Proj_H, [3]), dim=3), [3]) > 0.5).float()
        cs_m3_L = (torch.flip(torch.cumsum(torch.flip(m3, [3]), dim=3), [3]) > 0.5).float()
        cs_proj_R = (torch.cumsum(Proj_H, dim=3) > 0.5).float()
        Path3_H = torch.clamp(cs_m3_R * cs_proj_L + cs_m3_L * cs_proj_R + m3 + Proj_H, 0, 1) * Ray3
        
        Path_Ray3 = is_vert3 * Path3_V + is_horiz3 * Path3_H
        
        Proj = is_vert3 * Proj_V + is_horiz3 * Proj_H
        
        # Trim Ray2
        cs_m2_D = (torch.cumsum(m2, dim=2) > 0.5).float()
        cs_CI_U = (torch.flip(torch.cumsum(torch.flip(Chosen_Intersect, [2]), dim=2), [2]) > 0.5).float()
        cs_m2_U = (torch.flip(torch.cumsum(torch.flip(m2, [2]), dim=2), [2]) > 0.5).float()
        cs_CI_D = (torch.cumsum(Chosen_Intersect, dim=2) > 0.5).float()
        Path2_V = torch.clamp(cs_m2_D * cs_CI_U + cs_m2_U * cs_CI_D + m2 + Chosen_Intersect, 0, 1) * Ray2
        
        cs_m2_R = (torch.cumsum(m2, dim=3) > 0.5).float()
        cs_CI_L = (torch.flip(torch.cumsum(torch.flip(Chosen_Intersect, [3]), dim=3), [3]) > 0.5).float()
        cs_m2_L = (torch.flip(torch.cumsum(torch.flip(m2, [3]), dim=3), [3]) > 0.5).float()
        cs_CI_R = (torch.cumsum(Chosen_Intersect, dim=3) > 0.5).float()
        Path2_H = torch.clamp(cs_m2_R * cs_CI_L + cs_m2_L * cs_CI_R + m2 + Chosen_Intersect, 0, 1) * Ray2
        
        Path_Ray2 = is_vert2 * Path2_V + is_horiz2 * Path2_H
        
        Final_Path = torch.clamp(Path_Ray3 + Path_Perp + Path_Ray2, 0, 1)
        
        out_colors = torch.zeros_like(x)
        out_colors[:, 3:4, :, :] = Final_Path
        
        # Only overwrite 0s
        mask_0 = x[:, 0:1, :, :]
        overwrite = Final_Path * mask_0
        
        out_colors[:, 3:4, :, :] = overwrite
        # Keep everything else
        out_colors = out_colors + x * (1.0 - overwrite)
        
        self.last_ci = Chosen_Intersect
        self.last_proj = Proj
        self.last_ray2 = Ray2
        self.last_corr3 = Corridor3
        self.last_int = Intersect
        self.last_vd3 = Valid_Dist3
        
        return out_colors

def test():
    with open("neurogolf/data/raw/task066.json") as f:
        data = json.load(f)
    
    net = TaskNet066()
    net.eval()
    
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        # To tensor
        h, w = inp.shape
        x = torch.zeros((1, 10, h, w))
        for r in range(h):
            for c in range(w):
                x[0, int(inp[r,c]), r, c] = 1.0
                
        pred = net(x)
        pred_grid = torch.argmax(pred, dim=1)[0].numpy()
        
        match = np.array_equal(pred_grid, out)
        print(f"Train {i}: {match}")
        if i == 0:
            print("Ray2 at 7,17:", net.last_ray2.numpy()[0,0,7,17])
            print("Corridor3 at 7,17:", net.last_corr3.numpy()[0,0,7,17])
            print("Intersect at 7,17:", net.last_int.numpy()[0,0,7,17])
            print("Valid_Dist3 at 6,17:", net.last_vd3.numpy()[0,0,6,17])
            print("Valid_Dist3 at 7,17:", net.last_vd3.numpy()[0,0,7,17])
        if not match:
            # find chosen intersect
            print("Chosen Intersect:")
            print(np.argwhere(net.last_ci.numpy()[0,0]))
            print("Proj:")
            print(np.argwhere(net.last_proj.numpy()[0,0]))
            diff = (pred_grid != out)
            print(np.argwhere(diff))

if __name__ == "__main__":
    test()
