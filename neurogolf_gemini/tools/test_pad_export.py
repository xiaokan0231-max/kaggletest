import torch
import torch.nn.functional as F

class PadModel(torch.nn.Module):
    def forward(self, x):
        _, _, H, W = x.shape
        x_pad = F.pad(x, (0, 30 - W, 0, 30 - H))
        grid = torch.argmax(x_pad, dim=1, keepdim=True).float()
        
        out = grid + 1.0
        return out[:, :, :H, :W]

model = PadModel()
dummy = torch.randn(1, 10, 10, 14)
torch.onnx.export(
    model, 
    dummy, 
    "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/test_pad.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        'input': {0: 'batch_size', 2: 'height', 3: 'width'},
        'output': {0: 'batch_size', 2: 'height', 3: 'width'}
    },
    opset_version=17
)
print("Success")
