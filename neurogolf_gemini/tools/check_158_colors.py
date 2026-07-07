import json
import numpy as np

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

for ex in task_data['train'] + task_data.get('test', []):
    inp = np.array(ex['input'])
    colors, counts = np.unique(inp, return_counts=True)
    bg = colors[np.argmax(counts)]
    non_bg = [c for c in colors if c != bg]
    print(f"Non-bg colors: {non_bg}")
