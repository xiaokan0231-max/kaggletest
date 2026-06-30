import json
import numpy as np

def bfs_shortest_path(inp, start_color=3, target_color=2, obstacle=8):
    h, w = inp.shape
    start_mask = (inp == start_color)
    target_mask = (inp == target_color)
    
    # Distance from start
    dist = np.full((h, w), 999, dtype=int)
    dist[start_mask] = 0
    
    q = list(zip(*np.where(start_mask)))
    dirs = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # BFS
    while q:
        r, c = q.pop(0)
        d = dist[r, c]
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                if inp[nr, nc] != obstacle and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    q.append((nr, nc))
                    
    # Distance from target
    dist_t = np.full((h, w), 999, dtype=int)
    dist_t[target_mask] = 0
    q = list(zip(*np.where(target_mask)))
    while q:
        r, c = q.pop(0)
        d = dist_t[r, c]
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                if inp[nr, nc] != obstacle and dist_t[nr, nc] > d + 1:
                    dist_t[nr, nc] = d + 1
                    q.append((nr, nc))
                    
    # The shortest path length between start and target
    min_len = np.min(dist[target_mask])
    
    # All pixels on ANY shortest path
    on_path = (dist + dist_t == min_len)
    
    # Fill with start color
    out = inp.copy()
    out[on_path & (inp == 0)] = start_color
    return out

def analyze_task(filepath):
    with open(filepath) as f:
        data = json.load(f)
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        pred = bfs_shortest_path(inp)
        match = np.array_equal(pred, out)
        print(f"Train {i}: Match = {match}")
        if not match:
            print("Predicted path pixels:")
            print((pred == 3).astype(int))
            print("Actual path pixels:")
            print((out == 3).astype(int))

if __name__ == "__main__":
    analyze_task("neurogolf/data/raw/task066.json")
