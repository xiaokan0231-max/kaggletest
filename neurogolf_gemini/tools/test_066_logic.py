import json
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task066.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train']):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    
    # Colors
    start_c = 3
    target_c = 2
    obs_c = 8
    
    # Find start pixels
    sy, sx = np.where(inp == start_c)
    
    # Start is a segment. Let's find the two tips and their outward directions
    tips = []
    for y, x in zip(sy, sx):
        # check neighbors
        neighbors = []
        for dy, dx, d_idx in [(-1, 0, 3), (1, 0, 1), (0, -1, 2), (0, 1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and inp[ny, nx] == start_c:
                neighbors.append(d_idx)
        if len(neighbors) == 1:
            # Tip! Outward direction is opposite of neighbor
            out_dir = (neighbors[0] + 2) % 4
            tips.append((y, x, out_dir))
            
    # Calculate free space for each tip
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
            
        # Tie breaker: does it move towards target?
        # A simple score: dist * 10 + (1 if moving towards target else 0)
        # Wait, Ex 1 dists are both 1.
        if d == 0: moving_towards = target_x_center > x
        elif d == 1: moving_towards = target_y_center > y
        elif d == 2: moving_towards = target_x_center < x
        else: moving_towards = target_y_center < y
        
        score = dist * 10 + (1 if moving_towards else 0)
        
        if score > best_score:
            best_score = score
            best_tip = (y, x, d)
            
    pos_y, pos_x, curr_dir = best_tip
    
    out_pred = inp.copy()
    
    # Ray casting
    for segment in range(4):
        dy, dx = [ (0, 1), (1, 0), (0, -1), (-1, 0) ][curr_dir]
        
        # dist to obs
        cy, cx = pos_y, pos_x
        d_obs = 0
        while True:
            cy += dy
            cx += dx
            if not (0 <= cy < H and 0 <= cx < W): break
            if inp[cy, cx] == obs_c: break
            d_obs += 1
            
        # dist to align
        d_align = 1000
        if curr_dir in [0, 2]: # Horizontal
            # target_x bounds
            tx_min, tx_max = tx.min(), tx.max()
            if curr_dir == 0 and tx_min > pos_x: d_align = tx_min - pos_x
            elif curr_dir == 2 and tx_max < pos_x: d_align = pos_x - tx_max
        else: # Vertical
            ty_min, ty_max = ty.min(), ty.max()
            if curr_dir == 1 and ty_min > pos_y: d_align = ty_min - pos_y
            elif curr_dir == 3 and ty_max < pos_y: d_align = pos_y - ty_max
            
        step = min(d_obs, d_align)
        
        if step == 0 and d_align != 0:
            # Reached obstacle without aligning! Wait, this means we must turn early?
            pass
            
        # Draw line
        for _ in range(step):
            pos_y += dy
            pos_x += dx
            out_pred[pos_y, pos_x] = start_c
            
        # Check if reached target
        if out_pred[pos_y, pos_x] == target_c or np.any((ty == pos_y) & (tx == pos_x)):
            # Reached! (wait, we overwrite target_c? No, the original drawing doesn't overwrite 2)
            break
            
        # Determine next dir
        if curr_dir in [0, 2]:
            if target_y_center > pos_y: curr_dir = 1
            else: curr_dir = 3
        else:
            if target_x_center > pos_x: curr_dir = 0
            else: curr_dir = 2

    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
