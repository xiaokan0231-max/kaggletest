import onnxruntime
import numpy as np
import json

path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task028.onnx"
session = onnxruntime.InferenceSession(path)

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task028.json") as f:
    data = json.load(f)
    
for idx, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])

    padded = np.zeros((30, 30), dtype=np.int64)
    padded[:inp.shape[0], :inp.shape[1]] = inp

    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for i in range(10):
        x[0, i] = (padded == i)

    y = session.run(None, {'input': x})[0]
    y_argmax = np.argmax(y[0], axis=0)

    res = y_argmax[:inp.shape[0], :inp.shape[1]]
    if not np.array_equal(res, out):
        print(f"Failed Train {idx}")
        print("EXPECTED:", out)
        print("ACTUAL:", res)
    else:
        print(f"Passed Train {idx}")

print("INPUT:")
print(inp)
