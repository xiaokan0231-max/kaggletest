import json
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task066.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train']):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    start_c = 3
    target_c = 2
    obs_c = 8
    
    sy, sx = np.where(inp == start_c)
    tips = []
    for y, x in zip(sy, sx):
        neighbors = []
        for dy, dx, d_idx in [(-1, 0, 3), (1, 0, 1), (0, -1, 2), (0, 1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and inp[ny, nx] == start_c:
                neighbors.append(d_idx)
        if len(neighbors) == 1:
            out_dir = (neighbors[0] + 2) % 4
            tips.append((y, x, out_dir))
            
    best_tip = None
    best_score = -1
    
    ty, tx = np.where(inp == target_c)
    target_y_center = np.mean(ty)
    target_x_center = np.mean(tx)
    
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
            
    pos_y, pos_x, curr_dir = best_tip
    print(f"Ex {i} start: ({pos_y}, {pos_x}), dir: {curr_dir}")
    
    out_pred = inp.copy()
    
    for segment in range(4):
        dy, dx = [ (0, 1), (1, 0), (0, -1), (-1, 0) ][curr_dir]
        
        cy, cx = pos_y, pos_x
        d_obs = 0
        while True:
            cy += dy
            cx += dx
            if not (0 <= cy < H and 0 <= cx < W): break
            if inp[cy, cx] == obs_c: break
            d_obs += 1
            
        d_align = 1000
        if curr_dir in [0, 2]:
            tx_min, tx_max = tx.min(), tx.max()
            if curr_dir == 0 and tx_min > pos_x: d_align = tx_min - pos_x
            elif curr_dir == 2 and tx_max < pos_x: d_align = pos_x - tx_max
        else:
            ty_min, ty_max = ty.min(), ty.max()
            if curr_dir == 1 and ty_min > pos_y: d_align = ty_min - pos_y
            elif curr_dir == 3 and ty_max < pos_y: d_align = pos_y - ty_max
            
        step = min(d_obs, d_align)
        print(f"  Seg {segment}: dir {curr_dir}, d_obs {d_obs}, d_align {d_align}, step {step}")
        
        for _ in range(step):
            pos_y += dy
            pos_x += dx
            if out_pred[pos_y, pos_x] != target_c:
                out_pred[pos_y, pos_x] = start_c
            
        if np.any((ty == pos_y) & (tx == pos_x)):
            print("  Reached target!")
            break
            
        if curr_dir in [0, 2]:
            if target_y_center > pos_y: curr_dir = 1
            else: curr_dir = 3
        else:
            if target_x_center > pos_x: curr_dir = 0
            else: curr_dir = 2

    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
    if not np.array_equal(out_pred, out_expected):
        diff = np.where(out_pred != out_expected)
        for dy, dx in zip(*diff):
            print(f"  diff at ({dy}, {dx}): pred {out_pred[dy, dx]}, exp {out_expected[dy, dx]}")
