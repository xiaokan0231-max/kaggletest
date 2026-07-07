import onnxruntime as ort
import numpy as np
import json

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json') as f:
    data = json.load(f)
    
ex = data['train'][0]
inp = np.array(ex['input'])

B = 1
padded = np.zeros((B, 10, 30, 30), dtype=np.float32)
H, W = inp.shape
for i in range(H):
    for j in range(W):
        padded[0, inp[i, j], i, j] = 1.0

sess = ort.InferenceSession('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task006.onnx')
onnx_out = sess.run(None, {'input': padded})[0]

pred_colors = np.argmax(onnx_out[0], axis=0)
print(pred_colors)
