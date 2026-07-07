import onnxruntime
import numpy as np
import json

path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task024.onnx"
session = onnxruntime.InferenceSession(path)

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task024.json") as f:
    data = json.load(f)
    
ex = data['train'][0]
inp = np.array(ex['input'])
out = np.array(ex['output'])

# Pad to 30x30
padded = np.zeros((30, 30), dtype=np.int64)
padded[:inp.shape[0], :inp.shape[1]] = inp

# Convert to one-hot
x = np.zeros((1, 10, 30, 30), dtype=np.float32)
for i in range(10):
    x[0, i] = (padded == i)

y = session.run(None, {'input': x})[0]
y_argmax = np.argmax(y[0], axis=0)

res = y_argmax[:inp.shape[0], :inp.shape[1]]

print("EXPECTED:")
print(out)
print("ACTUAL:")
print(res)

print("INPUT:")
print(inp)
