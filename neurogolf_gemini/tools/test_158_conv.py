import json
import numpy as np
import scipy.signal

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    H, W = inp.shape
    
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    conn_color = None
    for c in non_bg:
        mask = (inp == c)
        dilated = np.pad(mask, 1, mode='constant')[1:-1, 1:-1]
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                dilated |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        touches = set(inp[dilated]) - {c, bg}
        if len(touches) == 2:
            conn_color = c
            break
            
    endpoints = [c for c in non_bg if c != conn_color]
    c1, c2 = endpoints
    
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
    
    out_pred = inp.copy()
    
    c1_mask = (inp == c1)
    c2_mask = (inp == c2)
    
    T_conn = np.argwhere(conn_mask)
    
    out_mask = np.zeros_like(inp, dtype=bool)
    
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
                    # Filter by size S
                    kernel = np.ones((S, S))
                    conv = scipy.signal.convolve2d(base_c1.astype(float), kernel, mode='valid')
                    valid_centers = (conv == S * S)
                    
                    if np.any(valid_centers):
                        # Reconstruct the valid base_c1 pixels
                        valid_base_c1 = np.zeros_like(base_c1)
                        y_valid, x_valid = np.where(valid_centers)
                        for yv, xv in zip(y_valid, x_valid):
                            valid_base_c1[yv:yv+S, xv:xv+S] = True
                            
                        for y_c, x_c in T_conn:
                            dy_c = y_c - y1_t
                            dx_c = x_c - x1_t
                            
                            DY_c = dy_c * sign_y * S
                            DX_c = dx_c * sign_x * S
                            
                            drawn = shift_mask(valid_base_c1, DY_c, DX_c)
                            out_mask |= drawn
                        
    out_pred[out_mask] = conn_color
    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
