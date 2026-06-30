import json
import torch
import torch.nn.functional as F
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task066.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train']):
    inp = torch.tensor(ex['input'], dtype=torch.long)
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    start_c = 3
    target_c = 2
    obs_c = 8
    
    # 1. Target Center
    target_mask = (inp == target_c).float()
    Y_grid, X_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    target_y_center = (target_mask * Y_grid).sum() / target_mask.sum()
    target_x_center = (target_mask * X_grid).sum() / target_mask.sum()
    
    # 2. Tips
    start_mask = (inp == start_c).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor([[[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]]])
    neighbors = F.conv2d(start_mask, kernel, padding=1).squeeze(0).squeeze(0)
    
    tips_mask = ((inp == start_c) & (neighbors == 1)).float()
    
    # Extract top 2 tips
    flat_tips = tips_mask.flatten()
    vals, indices = torch.topk(flat_tips, 2)
    
    # Compute scores for both tips
    best_score = torch.tensor(-1000.0)
    best_y = torch.tensor(0)
    best_x = torch.tensor(0)
    best_dir = torch.tensor(0)
    
    pad_inp = F.pad(inp, (1, 1, 1, 1), value=obs_c)
    Y_pad, X_pad = torch.meshgrid(torch.arange(H+2), torch.arange(W+2), indexing='ij')
    
    for k in range(2):
        idx = indices[k]
        y = idx // W
        x = idx % W
        
        y_pad = y + 1
        x_pad = x + 1
        
        # Check directions (0: R, 1: D, 2: L, 3: U)
        is_valid = vals[k] > 0.5
        
        n_R = pad_inp[y_pad, x_pad+1] == start_c
        n_D = pad_inp[y_pad+1, x_pad] == start_c
        n_L = pad_inp[y_pad, x_pad-1] == start_c
        n_U = pad_inp[y_pad-1, x_pad] == start_c
        
        # Tip direction is opposite to neighbor
        dir_k = torch.where(n_L, torch.tensor(0),
                torch.where(n_U, torch.tensor(1),
                torch.where(n_R, torch.tensor(2), torch.tensor(3))))
                
        # dist
        mask0 = (Y_pad == y_pad) & (X_pad > x_pad) & (pad_inp == obs_c)
        d0 = torch.where(mask0, X_pad, torch.tensor(1000)).min() - x_pad - 1
        
        mask1 = (X_pad == x_pad) & (Y_pad > y_pad) & (pad_inp == obs_c)
        d1 = torch.where(mask1, Y_pad, torch.tensor(1000)).min() - y_pad - 1
        
        mask2 = (Y_pad == y_pad) & (X_pad < x_pad) & (pad_inp == obs_c)
        d2 = x_pad - torch.where(mask2, X_pad, torch.tensor(-1000)).max() - 1
        
        mask3 = (X_pad == x_pad) & (Y_pad < y_pad) & (pad_inp == obs_c)
        d3 = y_pad - torch.where(mask3, Y_pad, torch.tensor(-1000)).max() - 1
        
        dist = torch.where(dir_k == 0, d0,
               torch.where(dir_k == 1, d1,
               torch.where(dir_k == 2, d2, d3)))
               
        moving_towards = torch.where(dir_k == 0, target_x_center > x,
                         torch.where(dir_k == 1, target_y_center > y,
                         torch.where(dir_k == 2, target_x_center < x, target_y_center < y))).float()
                         
        score = dist.float() * 10.0 + moving_towards
        score = torch.where(is_valid, score, torch.tensor(-1000.0))
        
        is_better = score > best_score
        best_score = torch.where(is_better, score, best_score)
        best_y = torch.where(is_better, y, best_y)
        best_x = torch.where(is_better, x, best_x)
        best_dir = torch.where(is_better, dir_k, best_dir)
        
    pos_y = best_y + 1
    pos_x = best_x + 1
    curr_dir = best_dir
    
    out_pred = pad_inp.clone()
    reached_target = torch.tensor(False)
    
    for segment in range(4):
        mask0 = (Y_pad == pos_y) & (X_pad > pos_x) & ((pad_inp == obs_c) | (pad_inp == target_c))
        X_obs0 = torch.where(mask0, X_pad, torch.tensor(1000))
        d_obs0 = X_obs0.min() - pos_x
        
        mask1 = (X_pad == pos_x) & (Y_pad > pos_y) & ((pad_inp == obs_c) | (pad_inp == target_c))
        Y_obs1 = torch.where(mask1, Y_pad, torch.tensor(1000))
        d_obs1 = Y_obs1.min() - pos_y
        
        mask2 = (Y_pad == pos_y) & (X_pad < pos_x) & ((pad_inp == obs_c) | (pad_inp == target_c))
        X_obs2 = torch.where(mask2, X_pad, torch.tensor(-1000))
        d_obs2 = pos_x - X_obs2.max()
        
        mask3 = (X_pad == pos_x) & (Y_pad < pos_y) & ((pad_inp == obs_c) | (pad_inp == target_c))
        Y_obs3 = torch.where(mask3, Y_pad, torch.tensor(-1000))
        d_obs3 = pos_y - Y_obs3.max()
        
        step = torch.where(curr_dir == 0, d_obs0,
               torch.where(curr_dir == 1, d_obs1,
               torch.where(curr_dir == 2, d_obs2, d_obs3)))
               
        hit_y = torch.where(curr_dir == 1, pos_y + step,
                torch.where(curr_dir == 3, pos_y - step, pos_y))
        hit_x = torch.where(curr_dir == 0, pos_x + step,
                torch.where(curr_dir == 2, pos_x - step, pos_x))
                
        hit_mask = (Y_pad == hit_y) & (X_pad == hit_x)
        hit_c = pad_inp[hit_mask].max()
        
        is_target = (hit_c == target_c)
        step = torch.where(hit_c == obs_c, step - 1, step)
        
        step = torch.where(reached_target, torch.tensor(0), step)
        reached_target = reached_target | is_target
        
        draw0 = (Y_pad == pos_y) & (X_pad >= pos_x) & (X_pad <= pos_x + step)
        draw1 = (X_pad == pos_x) & (Y_pad >= pos_y) & (Y_pad <= pos_y + step)
        draw2 = (Y_pad == pos_y) & (X_pad <= pos_x) & (X_pad >= pos_x - step)
        draw3 = (X_pad == pos_x) & (Y_pad <= pos_y) & (Y_pad >= pos_y - step)
        
        draw_mask = torch.where(curr_dir == 0, draw0,
                    torch.where(curr_dir == 1, draw1,
                    torch.where(curr_dir == 2, draw2, draw3)))
                    
        out_pred = torch.where(draw_mask, torch.tensor(start_c), out_pred)
        
        pos_x = torch.where(curr_dir == 0, pos_x + step,
                torch.where(curr_dir == 2, pos_x - step, pos_x))
        pos_y = torch.where(curr_dir == 1, pos_y + step,
                torch.where(curr_dir == 3, pos_y - step, pos_y))
                
        target_y_c_pad = target_y_center + 1.0
        target_x_c_pad = target_x_center + 1.0
        
        next_dir_horiz = torch.where(target_y_c_pad > pos_y.float(), torch.tensor(1), torch.tensor(3))
        next_dir_vert = torch.where(target_x_c_pad > pos_x.float(), torch.tensor(0), torch.tensor(2))
        
        curr_dir = torch.where((curr_dir == 0) | (curr_dir == 2), next_dir_horiz, next_dir_vert)

    out_pred[pad_inp == target_c] = target_c
    out_final = out_pred[1:-1, 1:-1].numpy()
    
    print(f"Ex {i} Match: {np.array_equal(out_final, out_expected)}")
    if not np.array_equal(out_final, out_expected):
        diff = np.where(out_final != out_expected)
        for dy, dx in zip(*diff):
            print(f"  diff at ({dy}, {dx}): pred {out_final[dy, dx]}, exp {out_expected[dy, dx]}")
