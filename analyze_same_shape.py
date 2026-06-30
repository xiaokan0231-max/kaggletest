import json
import os
import glob
import numpy as np

files = [
    'task017.json', 'task018.json', 'task023.json', 
    'task027.json', 'task033.json', 'task034.json', 
    'task035.json', 'task037.json', 'task042.json', 'task044.json'
]

for fname in files:
    with open(f'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/{fname}') as f:
        data = json.load(f)
    print(f"--- {fname} ---")
    
    ex = data['train'][0]
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    print(f"Shape: {inp.shape} -> {out.shape}")
    
    diff = (inp != out)
    diff_count = np.sum(diff)
    print(f"Diff count: {diff_count} / {inp.size}")
    
    in_colors = np.unique(inp)
    out_colors = np.unique(out)
    print(f"In colors: {in_colors}, Out colors: {out_colors}")
    
    # Just show a small crop of the diff
    r, c = np.where(diff)
    if len(r) > 0:
        rmin, rmax = np.min(r), np.max(r)
        cmin, cmax = np.min(c), np.max(c)
        print(f"Diff bbox size: {rmax-rmin+1}x{cmax-cmin+1}")
        print("Diff values (Input -> Output):")
        for i in range(min(5, len(r))):
            print(f"  ({r[i]},{c[i]}): {inp[r[i],c[i]]} -> {out[r[i],c[i]]}")
    else:
        print("No difference!")
