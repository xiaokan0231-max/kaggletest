import numpy as np

def fit_and_generate(task_data, task_id):
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
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            return None

    bg_color = 0
    best_params = None
    
    for target_color in list(range(10)) + [-1]:
        match = True
        for ex in train_data:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if target_color == -1:
                mask = (inp != bg_color)
            else:
                mask = (inp == target_color)
                
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows):
                match = False
                break
                
            row_idx = np.where(rows)[0]
            col_idx = np.where(cols)[0]
            
            # The extracted output size is exactly the number of kept rows and cols
            if out.shape != (len(row_idx), len(col_idx)):
                match = False
                break
                
            # Extract
            crop = inp[np.ix_(row_idx, col_idx)]
            if not np.array_equal(crop, out):
                match = False
                break
                
        if match:
            best_params = target_color
            break
            
    if best_params is None:
        return None
        
    target_color = best_params
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_color = {target_color}
        self.bg_color = 0

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        B, C, H, W = x.shape
        MAX_DIM = 30
        
        if self.target_color == -1:
            bg_layer = x[:, self.bg_color:self.bg_color+1, :, :]
            mask = valid_mask - bg_layer
        else:
            mask = x[:, self.target_color:self.target_color+1, :, :]
            
        keep_row = torch.max(mask, dim=3, keepdim=True)[0] # [B, 1, MAX_DIM, 1]
        keep_col = torch.max(mask, dim=2, keepdim=True)[0] # [B, 1, 1, MAX_DIM]
        
        krw = keep_row.float().view(B, 1, 1, MAX_DIM)
        pr0 = torch.cumsum(krw, dim=3)
        pr = pr0 - 1.0
        iota_r = torch.arange(MAX_DIM, device=x.device, dtype=torch.float32).view(1, 1, MAX_DIM, 1)
        ErB = (iota_r == pr)
        Sr = ErB.float() * krw
        
        pc0 = torch.cumsum(keep_col.float(), dim=3)
        pcw = pc0 - 1.0
        pc = pcw.view(B, 1, MAX_DIM, 1)
        kc = keep_col.float().view(B, 1, MAX_DIM, 1)
        iota_c = torch.arange(MAX_DIM, device=x.device, dtype=torch.float32).view(1, 1, 1, MAX_DIM)
        EcB = (pc == iota_c)
        T = EcB.float() * kc
        
        out_colors = torch.matmul(torch.matmul(Sr, x), T)
        
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
        dynamic_axes={{'input': {{0: 'batch_size'}},
                      'output': {{0: 'batch_size'}}}}
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
