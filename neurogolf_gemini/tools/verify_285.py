import json
import numpy as np
import onnxruntime as ort

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/arc-agi_training_challenges.json") as f:
    data = json.load(f)

task_id = "d0f5fe59" # 285 in training set? Wait, d0f5fe59 is just a random id.
# Let's find the task with the examples we used.
task_data = None
for k, v in data.items():
    ex = v['train'][0]
    if len(ex['input']) == 20 and len(ex['input'][0]) == 19 and ex['input'][4][5] == 2 and ex['input'][4][6] == 8:
        task_id = k
        task_data = v
        break
    # Wait, the input size of ex0 is 20x19 in my inspect output!
    # Let's just search by size.
    if len(ex['input']) == 20 and len(ex['input'][0]) == 19:
        if sum(row.count(8) for row in ex['input']) > 0:
            task_id = k
            task_data = v

if task_data is None:
    print("Task not found in training, let's check evaluation")
    with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/arc-agi_evaluation_challenges.json") as f:
        data = json.load(f)
    for k, v in data.items():
        ex = v['train'][0]
        if len(ex['input']) == 20 and len(ex['input'][0]) == 19:
            task_id = k
            task_data = v
            break

if task_data is None:
    print("Could not find task")
    exit(1)

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
            # print diff
            print(f"Pred shape: {pred.shape}, Out shape: {out.shape}")

print("All Correct:", all_correct)
