import json
import torch
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
    
    # Precompute start
    sy, sx = torch.where(inp == start_c)
    tips = []
    for y, x in zip(sy, sx):
        neighbors = []
        for dy, dx, d_idx in [(-1, 0, 3), (1, 0, 1), (0, -1, 2), (0, 1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and inp[ny, nx] == start_c:
                neighbors.append(d_idx)
        if len(neighbors) == 1:
            out_dir = (neighbors[0] + 2) % 4
            tips.append((y.item(), x.item(), out_dir))
            
    ty, tx = torch.where(inp == target_c)
    target_y_center = ty.float().mean()
    target_x_center = tx.float().mean()
    
    best_tip = None
    best_score = -1
    for y, x, d in tips:
        dy, dx = [ (0, 1), (1, 0), (0, -1), (-1, 0) ][d]
        cy, cx = y, x
        dist = 0
        while True:
            cy += dy
            cx += dx
            if not (0 <= cy < H and 0 <= cx < W): break
            if inp[cy, cx] == obs_c: break
            dist += 1
            
        if d == 0: moving_towards = target_x_center > x
        elif d == 1: moving_towards = target_y_center > y
        elif d == 2: moving_towards = target_x_center < x
        else: moving_towards = target_y_center < y
        
        score = dist * 10 + (1 if moving_towards else 0)
        if score > best_score:
            best_score = score
            best_tip = (y, x, d)
            
    start_y, start_x, start_dir = best_tip
    
    # Tensor logic
    pos_y = torch.tensor(start_y + 1, dtype=torch.long)
    pos_x = torch.tensor(start_x + 1, dtype=torch.long)
    curr_dir = torch.tensor(start_dir, dtype=torch.long)
    
    pad_inp = torch.nn.functional.pad(inp, (1, 1, 1, 1), value=obs_c)
    out_pred = pad_inp.clone()
    
    Y, X = torch.meshgrid(torch.arange(H+2), torch.arange(W+2), indexing='ij')
    
    for segment in range(4):
        mask0 = (Y == pos_y) & (X > pos_x) & ((pad_inp == obs_c) | (pad_inp == target_c))
        X_obs0 = torch.where(mask0, X, torch.tensor(1000))
        d_obs0 = X_obs0.min() - pos_x
        
        mask1 = (X == pos_x) & (Y > pos_y) & ((pad_inp == obs_c) | (pad_inp == target_c))
        Y_obs1 = torch.where(mask1, Y, torch.tensor(1000))
        d_obs1 = Y_obs1.min() - pos_y
        
        mask2 = (Y == pos_y) & (X < pos_x) & ((pad_inp == obs_c) | (pad_inp == target_c))
        X_obs2 = torch.where(mask2, X, torch.tensor(-1000))
        d_obs2 = pos_x - X_obs2.max()
        
        mask3 = (X == pos_x) & (Y < pos_y) & ((pad_inp == obs_c) | (pad_inp == target_c))
        Y_obs3 = torch.where(mask3, Y, torch.tensor(-1000))
        d_obs3 = pos_y - Y_obs3.max()
        
        step = torch.where(curr_dir == 0, d_obs0,
               torch.where(curr_dir == 1, d_obs1,
               torch.where(curr_dir == 2, d_obs2, d_obs3)))
               
        hit_y = torch.where(curr_dir == 1, pos_y + step,
                torch.where(curr_dir == 3, pos_y - step, pos_y))
        hit_x = torch.where(curr_dir == 0, pos_x + step,
                torch.where(curr_dir == 2, pos_x - step, pos_x))
                
        hit_mask = (Y == hit_y) & (X == hit_x)
        hit_c = pad_inp[hit_mask].max() # Should be exactly 1 element
        
        step = torch.where(hit_c == obs_c, step - 1, step)
        
        draw0 = (Y == pos_y) & (X >= pos_x) & (X <= pos_x + step)
        draw1 = (X == pos_x) & (Y >= pos_y) & (Y <= pos_y + step)
        draw2 = (Y == pos_y) & (X <= pos_x) & (X >= pos_x - step)
        draw3 = (X == pos_x) & (Y <= pos_y) & (Y >= pos_y - step)
        
        draw_mask = torch.where(curr_dir == 0, draw0,
                    torch.where(curr_dir == 1, draw1,
                    torch.where(curr_dir == 2, draw2, draw3)))
                    
        out_pred = torch.where(draw_mask, torch.tensor(start_c), out_pred)
        
        pos_x = torch.where(curr_dir == 0, pos_x + step,
                torch.where(curr_dir == 2, pos_x - step, pos_x))
        pos_y = torch.where(curr_dir == 1, pos_y + step,
                torch.where(curr_dir == 3, pos_y - step, pos_y))
                
        target_y_c_pad = target_y_center + 1
        target_x_c_pad = target_x_center + 1
        
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
