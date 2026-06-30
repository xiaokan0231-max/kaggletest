import json
import numpy as np

def extend_segment(inp, color):
    mask = (inp == color)
    coords = np.argwhere(mask)
    if len(coords) == 0: return None
    
    r1, c1 = coords[0]
    r2, c2 = coords[1]
    
    is_vertical = (c1 == c2)
    segment = []
    
    if is_vertical:
        c = c1
        # extend down
        r = max(r1, r2)
        while r < inp.shape[0] and inp[r, c] != 8:
            segment.append((r, c))
            r += 1
        # extend up
        r = min(r1, r2)
        while r >= 0 and inp[r, c] != 8:
            segment.append((r, c))
            r -= 1
    else:
        r = r1
        # extend right
        c = max(c1, c2)
        while c < inp.shape[1] and inp[r, c] != 8:
            segment.append((r, c))
            c += 1
        # extend left
        c = min(c1, c2)
        while c >= 0 and inp[r, c] != 8:
            segment.append((r, c))
            c -= 1
            
    return set(segment), is_vertical

def analyze_task(filepath):
    with open(filepath) as f:
        data = json.load(f)
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        seg3, vert3 = extend_segment(inp, 3)
        seg2, vert2 = extend_segment(inp, 2)
        
        print(f"Train {i}")
        
        actual_path = set(map(tuple, np.argwhere((out == 3) & (inp != 3))))
        print(f"Actual path size: {len(actual_path)}")
        
        # Check if actual path is subset of seg3 + seg2 + one perp line
        # Just print out the segments to see
        # print("Seg3:", seg3)
        # print("Seg2:", seg2)
        
        # Find intersections
        for r, c in actual_path:
            if (r, c) not in seg3 and (r, c) not in seg2:
                print(f"Perp pixel: {r, c}")
        
if __name__ == "__main__":
    analyze_task("neurogolf/data/raw/task066.json")
