import torch
from make_task285 import Task285
import json
import numpy as np

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

ex = data['train'][0]
inp = np.array(ex['input'])
H, W = inp.shape

model = Task285()
model.eval()

x = torch.zeros(1, 10, H, W)
for c in range(10):
    x[0, c] = torch.tensor(inp == c, dtype=torch.float32)

torch.onnx.export(
    model,
    x,
    "task285.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    }
)
print("Exported task285.onnx successfully!")
