import json
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json', 'r') as f:
    data = json.load(f)
    
for ex in data['train']:
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    H, W = inp.shape
    assert W % 2 == 1
    half = W // 2
    left = inp[:, :half]
    right = inp[:, half+1:]
    
    mask_left = (left == 1)
    mask_right = (right == 1)
    
    mask_out = mask_left & mask_right
    
    expected_out = np.zeros_like(left)
    expected_out[mask_out] = 2
    
    assert np.array_equal(expected_out, out)
    print("Match!")
