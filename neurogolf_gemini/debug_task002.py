import torch
import json
import numpy as np
from make_task002 import Task002
import scipy.ndimage

def solve_numpy(X):
    X = np.array(X)
    H, W = X.shape
    grid30 = np.full((30, 30), -1, dtype=int)
    grid30[:H, :W] = X
    boundary = (grid30 == 3)
    state = np.zeros((30, 30), dtype=bool)
    state[0, :] = ~boundary[0, :]
    state[-1, :] = ~boundary[-1, :]
    state[:, 0] = ~boundary[:, 0]
    state[:, -1] = ~boundary[:, -1]
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for _ in range(30):
        state = scipy.ndimage.binary_dilation(state, structure=struct) & (~boundary)
    inside = ~(state | boundary)
    inside_cropped = inside[:H, :W]
    out = X.copy()
    out[inside_cropped] = 4
    return out.tolist()

def convert_to_numpy(grid):
    benchmark = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            benchmark[0, color, r, c] = 1.0
    return benchmark

def convert_from_numpy(benchmark, H, W):
    example = []
    for row in range(H):
        cells = []
        for col in range(W):
            colors = [c for c in range(10) if benchmark[0, c, row, col] > 0.0]
            cells.append(colors[0] if len(colors) == 1 else (11 if colors else 10))
        example.append(cells)
    return example

def debug():
    model = Task002()
    model.eval()
    
    with open('../neurogolf/data/raw/task002.json') as f:
        data = json.load(f)
        
    for i, ex in enumerate(data['arc-gen']):
        inp = ex['input']
        H, W = len(inp), len(inp[0])
        
        # Numpy
        out_np = solve_numpy(inp)
        
        # PyTorch
        t_in = torch.tensor(convert_to_numpy(inp))
        with torch.no_grad():
            t_out = model(t_in).numpy()
        out_pt = convert_from_numpy(t_out, H, W)
        
        if out_np != out_pt:
            print(f"Mismatch at arc-gen {i}")
            print("Input:")
            for row in inp: print(row)
            print("Numpy:")
            for row in out_np: print(row)
            print("PyTorch:")
            for row in out_pt: print(row)
            break

if __name__ == "__main__":
    debug()
