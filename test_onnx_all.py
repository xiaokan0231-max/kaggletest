import onnxruntime as ort
import numpy as np
import json

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json') as f:
    data = json.load(f)
    
sess = ort.InferenceSession('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task006.onnx')

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    B = 1
    padded = np.zeros((B, 10, 30, 30), dtype=np.float32)
    H, W = inp.shape
    for r in range(H):
        for c in range(W):
            padded[0, inp[r, c], r, c] = 1.0
            
    onnx_out = sess.run(None, {'input': padded})[0]
    pred_colors = np.argmax(onnx_out[0], axis=0)
    
    # get the non-zero bbox of pred_colors (wait, verifier trims trailing zeros!)
    def get_bbox(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows):
            return None
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        return r_max+1, c_max+1

    # Verifier trims trailing ALL ZERO rows and columns
    # Actually, verify_network trims any row/col that is entirely zero at the END
    r_max = 29
    while r_max >= 0 and np.all(pred_colors[r_max, :] == 0):
        r_max -= 1
    c_max = 29
    while c_max >= 0 and np.all(pred_colors[:, c_max] == 0):
        c_max -= 1
        
    pred_cropped = pred_colors[:r_max+1, :c_max+1]
    
    print(f"Example {i}:")
    print("Expected:")
    print(out)
    print("Pred:")
    print(pred_cropped)
    
    if np.array_equal(out, pred_cropped):
        print("MATCH!")
    else:
        print("MISMATCH!")
