import torch
import torch.nn as nn
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out_colors = torch.zeros_like(x)
        out_colors[:, 0:1, :, :] = torch.max(out_colors[:, 0:1, :, :], x[:, 0:1, :, :])
        out_colors[:, 1:2, :, :] = torch.max(out_colors[:, 1:2, :, :], x[:, 1:2, :, :])
        out_colors[:, 2:3, :, :] = torch.max(out_colors[:, 2:3, :, :], x[:, 2:3, :, :])
        out_colors[:, 3:4, :, :] = torch.max(out_colors[:, 3:4, :, :], x[:, 3:4, :, :])
        out_colors[:, 4:5, :, :] = torch.max(out_colors[:, 4:5, :, :], x[:, 4:5, :, :])
        out_colors[:, 5:6, :, :] = torch.max(out_colors[:, 5:6, :, :], x[:, 5:6, :, :])
        out_colors[:, 2:3, :, :] = torch.max(out_colors[:, 2:3, :, :], x[:, 6:7, :, :])
        out_colors[:, 7:8, :, :] = torch.max(out_colors[:, 7:8, :, :], x[:, 7:8, :, :])
        out_colors[:, 8:9, :, :] = torch.max(out_colors[:, 8:9, :, :], x[:, 8:9, :, :])
        out_colors[:, 9:10, :, :] = torch.max(out_colors[:, 9:10, :, :], x[:, 9:10, :, :])

        return out_colors * valid_mask

def export_onnx():
    net = TaskNet()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task276.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                      'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
    )
    import os
    import onnx
    onnx_model = onnx.load_model(onnx_path, load_external_data=True)
    
    for tensor in onnx_model.graph.initializer:
        if tensor.HasField("data_location"):
            tensor.ClearField("data_location")
        if len(tensor.external_data) > 0:
            del tensor.external_data[:]
            
    for node in onnx_model.graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                if attr.t.HasField("data_location"):
                    attr.t.ClearField("data_location")
                if len(attr.t.external_data) > 0:
                    del attr.t.external_data[:]
            for t in attr.tensors:
                if t.HasField("data_location"):
                    t.ClearField("data_location")
                if len(t.external_data) > 0:
                    del t.external_data[:]
                    
    onnx.save_model(
        onnx_model,
        onnx_path,
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        size_threshold=1024*1024*1024
    )
    if os.path.exists(onnx_path + '.data'):
        os.remove(onnx_path + '.data')

if __name__ == '__main__':
    export_onnx()
