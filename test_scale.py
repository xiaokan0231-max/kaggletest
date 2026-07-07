import torch
import torch.nn as nn
import onnx

class ScaleNet(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

net = ScaleNet()
x = torch.zeros(1, 1, 5, 5)

torch.onnx.export(
    net,
    (x,),
    "test_scale.onnx",
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)
print("Exported successfully!")
