import numpy as np

def sim_cumsum(mask, axis, reverse):
    if axis == 2:
        if reverse:
            return np.cumsum(mask[::-1, :], axis=0)[::-1, :]
        else:
            return np.cumsum(mask, axis=0)
    else:
        if reverse:
            return np.cumsum(mask[:, ::-1], axis=1)[:, ::-1]
        else:
            return np.cumsum(mask, axis=1)

def sim_ray(start_mask, obs_mask, axis, reverse):
    cs_start = sim_cumsum(start_mask, axis, reverse)
    ray_fwd = (cs_start > 0).astype(int)
    obs_after = obs_mask & ray_fwd
    cs_obs = sim_cumsum(obs_after, axis, reverse)
    return ray_fwd & (cs_obs == 0)

def get_dist(ray_mask, axis, reverse):
    return sim_cumsum(ray_mask, axis, reverse) * ray_mask

def solve(inp):
    M3 = (inp == 3).astype(int)
    M2 = (inp == 2).astype(int)
    M8 = (inp == 8).astype(int)
    
    # Check orientation (simplified for python)
    coords3 = np.argwhere(M3)
    is_vert3 = (coords3[0,1] == coords3[1,1]) if len(coords3)>1 else True
    
    coords2 = np.argwhere(M2)
    is_vert2 = (coords2[0,1] == coords2[1,1]) if len(coords2)>1 else True
    
    R3_D = sim_ray(M3, M8, 2, False)
    R3_U = sim_ray(M3, M8, 2, True)
    R3_R = sim_ray(M3, M8, 3, False)
    R3_L = sim_ray(M3, M8, 3, True)
    
    R2_D = sim_ray(M2, M8, 2, False)
    R2_U = sim_ray(M2, M8, 2, True)
    R2_R = sim_ray(M2, M8, 3, False)
    R2_L = sim_ray(M2, M8, 3, True)
    
    if is_vert3:
        Ray3 = R3_D | R3_U
        Dist3 = get_dist(R3_D, 2, False) + get_dist(R3_U, 2, True)
        Corr_H = sim_ray(Ray3, M8, 3, False) | sim_ray(Ray3, M8, 3, True)
        Corridor3 = Corr_H
    else:
        Ray3 = R3_R | R3_L
        Dist3 = get_dist(R3_R, 3, False) + get_dist(R3_L, 3, True)
        Corr_V = sim_ray(Ray3, M8, 2, False) | sim_ray(Ray3, M8, 2, True)
        Corridor3 = Corr_V
        
    if is_vert2:
        Ray2 = R2_D | R2_U
    else:
        Ray2 = R2_R | R2_L
        
    Intersect = Corridor3 & Ray2
    
    # Find max Dist3 in Intersect
    Valid_Dist3 = Dist3 * Intersect
    max_d = np.max(Valid_Dist3)
    Chosen_Intersect = (Valid_Dist3 == max_d) & (Intersect > 0)
    
    # Proj onto Ray3
    # Corridor from Chosen_Intersect back to Ray3
    if is_vert3:
        # Corridor is horizontal
        R_CI_R = sim_ray(Chosen_Intersect, M8, 3, False)
        R_CI_L = sim_ray(Chosen_Intersect, M8, 3, True)
        Path_Horiz = (R_CI_R | R_CI_L) & Corridor3
        Proj = Path_Horiz & Ray3
    else:
        # Corridor is vertical
        R_CI_D = sim_ray(Chosen_Intersect, M8, 2, False)
        R_CI_U = sim_ray(Chosen_Intersect, M8, 2, True)
        Path_Vert = (R_CI_D | R_CI_U) & Corridor3
        Proj = Path_Vert & Ray3
        
    # Trim Ray3
    if is_vert3:
        Path_Ray3 = (sim_cumsum(M3, 2, False)>0) & (sim_cumsum(Proj, 2, True)>0) | \
                    (sim_cumsum(M3, 2, True)>0) & (sim_cumsum(Proj, 2, False)>0) | M3 | Proj
        Path_Ray3 = Path_Ray3 & Ray3
    else:
        Path_Ray3 = (sim_cumsum(M3, 3, False)>0) & (sim_cumsum(Proj, 3, True)>0) | \
                    (sim_cumsum(M3, 3, True)>0) & (sim_cumsum(Proj, 3, False)>0) | M3 | Proj
        Path_Ray3 = Path_Ray3 & Ray3
        
    # Trim Ray2
    if is_vert2:
        Path_Ray2 = (sim_cumsum(M2, 2, False)>0) & (sim_cumsum(Chosen_Intersect, 2, True)>0) | \
                    (sim_cumsum(M2, 2, True)>0) & (sim_cumsum(Chosen_Intersect, 2, False)>0) | M2 | Chosen_Intersect
        Path_Ray2 = Path_Ray2 & Ray2
    else:
        Path_Ray2 = (sim_cumsum(M2, 3, False)>0) & (sim_cumsum(Chosen_Intersect, 3, True)>0) | \
                    (sim_cumsum(M2, 3, True)>0) & (sim_cumsum(Chosen_Intersect, 3, False)>0) | M2 | Chosen_Intersect
        Path_Ray2 = Path_Ray2 & Ray2
        
    if is_vert3:
        Final_Path = Path_Ray3 | Path_Horiz | Path_Ray2
    else:
        Final_Path = Path_Ray3 | Path_Vert | Path_Ray2
        
    out = inp.copy()
    out[(Final_Path > 0) & (inp == 0)] = 3
    return out

import json
def test():
    with open("neurogolf/data/raw/task066.json") as f:
        data = json.load(f)
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        pred = solve(inp)
        match = np.array_equal(pred, out)
        print(f"Train {i}: {match}")
        if not match:
            print("Pred:")
            print((pred == 3).astype(int))
            print("Out:")
            print((out == 3).astype(int))

if __name__ == "__main__":
    test()
