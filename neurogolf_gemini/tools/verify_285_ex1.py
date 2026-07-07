import json
import numpy as np
import onnxruntime as ort

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task285.json") as f:
    task_data = json.load(f)

sess = ort.InferenceSession("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task285_opt.onnx")

ex = task_data['train'][1]
inp = np.array(ex['input'], dtype=np.float32).reshape(1, 1, len(ex['input']), len(ex['input'][0]))
out = np.array(ex['output'])
pred = sess.run(None, {'input': inp})[0].squeeze()

print("PRED unique:", np.unique(pred))
print("OUT unique:", np.unique(out))
diff = (pred != out)
print("Diff sum:", diff.sum())
if diff.sum() > 0:
    for y in range(out.shape[0]):
        if diff[y].sum() > 0:
            print(f"Row {y} - PRED: {pred[y].astype(int)}")
            print(f"Row {y} - OUT : {out[y]}")
