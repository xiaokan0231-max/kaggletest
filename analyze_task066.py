import json
import numpy as np

def analyze_task(filepath):
    with open(filepath) as f:
        data = json.load(f)
    for i, ex in enumerate(data['train']):
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        diff = (inp != out)
        print(f"Train {i}")
        print(f"Shape: {inp.shape}")
        
        # What changed?
        for c in np.unique(out[diff]):
            mask = (out == c) & diff
            coords = np.argwhere(mask)
            print(f"Color {c} filled {len(coords)} pixels.")
            print(f"Input had colors: {np.unique(inp)}")
            print(f"Output has colors: {np.unique(out)}")
            
        print("Input:")
        print(inp)
        print("Output:")
        print(out)

if __name__ == "__main__":
    analyze_task("neurogolf/data/raw/task066.json")
