import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

s = ndimage.generate_binary_structure(2, 2)

for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'])
    out_expected = np.array(ex['output'])
    
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    
    conn_color = None
    for c in non_bg:
        if ndimage.label(inp == c, structure=s)[1] == 1:
            conn_color = c
            break
            
    endpoints = [c for c in non_bg if c != conn_color]
    c1, c2 = endpoints
    
    conn_mask = (inp == conn_color)
    conn_dilated = ndimage.binary_dilation(conn_mask, structure=s)
    
    # Template
    y1_t, x1_t = np.where((inp == c1) & conn_dilated)
    y1_t, x1_t = y1_t[0], x1_t[0]
    
    y2_t, x2_t = np.where((inp == c2) & conn_dilated)
    y2_t, x2_t = y2_t[0], x2_t[0]
    
    T_conn = np.argwhere(conn_mask)
    
    dy_t = y2_t - y1_t
    dx_t = x2_t - x1_t
    
    out_pred = inp.copy()
    
    # Targets
    labeled1, n1 = ndimage.label(inp == c1, structure=s)
    targets1 = []
    for comp in range(1, n1+1):
        mask = (labeled1 == comp)
        if not np.any(mask & conn_dilated):
            r, c = np.where(mask)
            targets1.append({'r': r.min(), 'c': c.min(), 'size': len(r)})
            
    labeled2, n2 = ndimage.label(inp == c2, structure=s)
    targets2 = []
    for comp in range(1, n2+1):
        mask = (labeled2 == comp)
        if not np.any(mask & conn_dilated):
            r, c = np.where(mask)
            targets2.append({'r': r.min(), 'c': c.min(), 'size': len(r)})
            
    for t1 in targets1:
        for t2 in targets2:
            if t1['size'] == t2['size']:
                S = int(np.round(np.sqrt(t1['size'])))
                DY_T = t2['r'] - t1['r']
                DX_T = t2['c'] - t1['c']
                
                if abs(DY_T) == abs(dy_t) * S and abs(DX_T) == abs(dx_t) * S:
                    # Match found!
                    sign_y = 1 if (DY_T * dy_t > 0) else -1 if dy_t != 0 else 1
                    sign_x = 1 if (DX_T * dx_t > 0) else -1 if dx_t != 0 else 1
                    
                    for r_c, c_c in T_conn:
                        dr = r_c - y1_t
                        dc = c_c - x1_t
                        
                        Y_c = t1['r'] + dr * sign_y * S
                        X_c = t1['c'] + dc * sign_x * S
                        
                        # Draw S x S block
                        out_pred[Y_c:Y_c+S, X_c:X_c+S] = conn_color
                        
    print(f"Ex {i} Match: {np.array_equal(out_pred, out_expected)}")
