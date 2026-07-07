import json
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    if i != 1: continue
    inp = np.array(ex['input'])
    
    H, W = inp.shape
    
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    # Identify conn_color
    conn_color = 1
    c1, c2 = 6, 2
    
    conn_mask = (inp == conn_color)
    dilated = np.pad(conn_mask, 1, mode='constant')[1:-1, 1:-1]
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            dilated |= np.roll(np.roll(conn_mask, dy, axis=0), dx, axis=1)
            
    y1_t, x1_t = np.where((inp == c1) & dilated)
    y1_t, x1_t = y1_t[0], x1_t[0]
    
    y2_t, x2_t = np.where((inp == c2) & dilated)
    y2_t, x2_t = y2_t[0], x2_t[0]
    
    dy_t = y2_t - y1_t
    dx_t = x2_t - x1_t
    
    c1_mask = (inp == c1)
    c2_mask = (inp == c2)
    
    def shift_mask(m, dy, dx):
        res = np.zeros_like(m)
        y_start = max(0, dy)
        y_end = min(H, H + dy)
        x_start = max(0, dx)
        x_end = min(W, W + dx)
        
        my_start = max(0, -dy)
        my_end = min(H, H - dy)
        mx_start = max(0, -dx)
        mx_end = min(W, W - dx)
        
        if y_start < y_end and x_start < x_end:
            res[y_start:y_end, x_start:x_end] = m[my_start:my_end, mx_start:mx_end]
        return res

    for S in range(1, 16):
        for sign_y in [-1, 1]:
            for sign_x in [-1, 1]:
                DY = dy_t * sign_y * S
                DX = dx_t * sign_x * S
                
                shifted_c2 = shift_mask(c2_mask, -DY, -DX)
                base_c1 = c1_mask & shifted_c2
                
                if np.any(base_c1):
                    # Check if base_c1 intersects the spurious region!
                    # We know out_pred had diff at (2, 8), (2, 11), (5, 5), (5, 11)
                    print(f"S={S}, sign_y={sign_y}, sign_x={sign_x}, DY={DY}, DX={DX}")
                    print("  base_c1:", list(zip(*np.where(base_c1))))
