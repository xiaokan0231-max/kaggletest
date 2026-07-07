import json
import numpy as np
import scipy.ndimage

def solve_numpy(X):
    X = np.array(X)
    H, W = X.shape
    
    # Simulate the ONNX 30x30 grid where padded region has no color (we'll use -1 to represent no color)
    grid30 = np.full((30, 30), -1, dtype=int)
    grid30[:H, :W] = X
    
    boundary = (grid30 == 3)
    state = np.zeros((30, 30), dtype=bool)
    
    # Initialize state at the edges of the 30x30 grid
    state[0, :] = ~boundary[0, :]
    state[-1, :] = ~boundary[-1, :]
    state[:, 0] = ~boundary[:, 0]
    state[:, -1] = ~boundary[:, -1]
        
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)
    for _ in range(30):
        state = scipy.ndimage.binary_dilation(state, structure=struct) & (~boundary)
        
    inside = ~(state | boundary)
    
    # Crop back to H, W
    inside_cropped = inside[:H, :W]
    out = X.copy()
    out[inside_cropped] = 4
    return out.tolist()

def verify():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'neurogolf', 'data', 'raw', 'task002.json')
    with open(file_path) as f:
        data = json.load(f)
    
    all_passed = True
    for split in ['train', 'test', 'arc-gen']:
        if split not in data: continue
        for i, ex in enumerate(data[split]):
            inp = ex['input']
            expected = ex['output']
            out = solve_numpy(inp)
            if out != expected:
                print(f"FAILED {split} example {i}")
                all_passed = False
            else:
                print(f"PASSED {split} example {i}")
    
    if all_passed:
        print("All Numpy tests passed!")

if __name__ == '__main__':
    verify()
