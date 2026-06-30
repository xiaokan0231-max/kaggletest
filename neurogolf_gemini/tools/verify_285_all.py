import json
import numpy as np
import onnxruntime as ort

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task285.json") as f:
    task_data = json.load(f)

sess = ort.InferenceSession("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task285_opt.onnx")

all_correct = True
for i, ex in enumerate(task_data['train'] + task_data.get('test', [])):
    inp = np.array(ex['input'], dtype=np.float32).reshape(1, 1, len(ex['input']), len(ex['input'][0]))
    out = np.array(ex.get('output', ex['input']))
    
    pred = sess.run(None, {'input': inp})[0].squeeze()
    
    if 'output' in ex:
        if np.array_equal(pred, out):
            print(f"Ex {i}: PASS")
        else:
            print(f"Ex {i}: FAIL")
            all_correct = False

print("All Correct:", all_correct)
