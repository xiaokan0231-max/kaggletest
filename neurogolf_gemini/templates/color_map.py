import numpy as np

def fit_and_generate(task_data, task_id):
    """
    Tries to fit a pure color mapping template to the task data.
    If it fits perfectly, returns a string containing the PyTorch code.
    Otherwise returns None.
    """
    mapping = {}
    valid_mapping = True
    
    # We only check 'train' examples to fit the rule
    if 'train' not in task_data:
        if isinstance(task_data, list):
            train_data = [d for d in task_data if 'output' in d]
        else:
            return None
    else:
        train_data = task_data['train']
        
    for ex in train_data:
        inp = np.array(ex['input'])
        out = np.array(ex['output'])
        
        if inp.shape != out.shape:
            return None
            
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                ci = int(inp[i,j])
                co = int(out[i,j])
                if ci not in mapping:
                    mapping[ci] = co
                elif mapping[ci] != co:
                    valid_mapping = False
                    break
            if not valid_mapping: break
        if not valid_mapping: break
        
    if not valid_mapping:
        return None
        
    code = f"""import torch
import torch.nn as nn
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        out_colors = torch.zeros_like(x)
"""
    for c_in in range(10):
        if c_in in mapping:
            c_out = mapping[c_in]
        else:
            c_out = c_in
        code += f"        out_colors[:, {c_out}:{c_out+1}, :, :] = torch.max(out_colors[:, {c_out}:{c_out+1}, :, :], x[:, {c_in}:{c_in+1}, :, :])\n"
        
    code += f"""
        return out_colors * valid_mask

def export_onnx():
    net = TaskNet()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/{task_id}.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14,
        dynamic_axes={{'input': {{0: 'batch_size', 2: 'height', 3: 'width'}},
                      'output': {{0: 'batch_size', 2: 'height', 3: 'width'}}}}
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
"""
    return code
