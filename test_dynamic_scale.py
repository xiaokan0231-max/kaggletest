import torch
import torch.nn as nn
import onnx

class DynamicScaleNet(nn.Module):
    def forward(self, x, factor):
        # factor is a 1D tensor [1]
        # repeat_interleave allows repeats as a tensor?
        y = torch.repeat_interleave(x, factor, dim=2)
        y = torch.repeat_interleave(y, factor, dim=3)
        return y

net = DynamicScaleNet()
x = torch.zeros(1, 1, 5, 5)
factor = torch.tensor([2], dtype=torch.int64)

torch.onnx.export(
    net,
    (x, factor),
    "test_dynamic_scale.onnx",
    opset_version=14,
    input_names=['input', 'factor'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)
print("Exported successfully!")
