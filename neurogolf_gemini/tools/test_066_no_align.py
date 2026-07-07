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
    
    pos_y = start_y
    pos_x = start_x
    curr_dir = start_dir
    
    out_pred = inp.numpy().copy()
    
    for segment in range(4):
        dy, dx = [ (0, 1), (1, 0), (0, -1), (-1, 0) ][curr_dir]
        
        step = 0
        cy, cx = pos_y, pos_x
        while True:
            n_cy = cy + dy
            n_cx = cx + dx
            if not (0 <= n_cy < H and 0 <= n_cx < W): break
            if inp[n_cy, n_cx] == obs_c: break
            if inp[n_cy, n_cx] == target_c: 
                step += 1
                break
            cy = n_cy
            cx = n_cx
            step += 1
            
        for _ in range(step):
            pos_y += dy
            pos_x += dx
            if out_pred[pos_y, pos_x] != target_c:
                out_pred[pos_y, pos_x] = start_c
            
        if out_pred[pos_y, pos_x] == target_c or inp[pos_y, pos_x] == target_c:
            break
            
        if curr_dir in [0, 2]:
            if target_y_center > pos_y: curr_dir = 1
            else: curr_dir = 3
        else:
            if target_x_center > pos_x: curr_dir = 0
            else: curr_dir = 2

    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
