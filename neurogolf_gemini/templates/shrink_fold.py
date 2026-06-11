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

    best_params = None
    
    for fold_axis in [0, 1]:
        for flip_second in [False, True]:
            for op in ['AND', 'OR', 'XOR', 'LEFT_NOT_RIGHT', 'RIGHT_NOT_LEFT']:
                for target_color in range(1, 10):
                    for output_color in range(1, 10):
                        match = True
                        for ex in train_data:
                            inp = np.array(ex['input'])
                            out = np.array(ex['output'])
                            H, W = inp.shape
                            
                            dim = H if fold_axis == 0 else W
                            
                            if dim % 2 == 1:
                                half = dim // 2
                                if fold_axis == 0:
                                    part1 = inp[:half, :]
                                    part2 = inp[dim-half:dim, :]
                                else:
                                    part1 = inp[:, :half]
                                    part2 = inp[:, dim-half:dim]
                            else:
                                half = dim // 2
                                if fold_axis == 0:
                                    part1 = inp[:half, :]
                                    part2 = inp[half:, :]
                                else:
                                    part1 = inp[:, :half]
                                    part2 = inp[:, half:]
                                    
                            if part1.shape != out.shape or part2.shape != out.shape:
                                match = False
                                break
                                
                            if flip_second:
                                part2 = np.flip(part2, axis=fold_axis)
                                
                            mask1 = (part1 == target_color)
                            mask2 = (part2 == target_color)
                            
                            if op == 'AND':
                                mask_out = mask1 & mask2
                            elif op == 'OR':
                                mask_out = mask1 | mask2
                            elif op == 'XOR':
                                mask_out = mask1 ^ mask2
                            elif op == 'LEFT_NOT_RIGHT':
                                mask_out = mask1 & (~mask2)
                            elif op == 'RIGHT_NOT_LEFT':
                                mask_out = mask2 & (~mask1)
                                
                            expected_out = np.zeros_like(part1)
                            expected_out[mask_out] = output_color
                            
                            if not np.array_equal(expected_out, out):
                                match = False
                                break
                                
                        if match:
                            best_params = (fold_axis, flip_second, op, target_color, output_color)
                            break
                    if best_params: break
                if best_params: break
            if best_params: break
        if best_params: break
        
    if best_params is None:
        return None
        
    fold_axis, flip_second, op, target_color, output_color = best_params
    
    code = f"""import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class TaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fold_axis = {fold_axis}
        self.flip_second = {flip_second}
        self.op = '{op}'
        self.target_color = {target_color}
        self.output_color = {output_color}

    def _crop_compact(self, content, keep_row, keep_col, B, MAX_DIM, flip_r=False, flip_c=False):
        krw = keep_row.float().view(B, 1, 1, MAX_DIM)
        pr0 = torch.cumsum(krw, dim=3)
        pr = pr0 - 1.0
        if flip_r:
            pr = krw.sum(dim=3, keepdim=True) - 1.0 - pr
            
        iota_r = torch.arange(MAX_DIM, device=content.device, dtype=torch.float32).view(1, 1, MAX_DIM, 1)
        ErB = (iota_r == pr)
        Sr = ErB.float() * krw
        
        pc0 = torch.cumsum(keep_col.float(), dim=3)
        pcw = pc0 - 1.0
        if flip_c:
            pcw = keep_col.float().sum(dim=3, keepdim=True) - 1.0 - pcw
            
        pc = pcw.view(B, 1, MAX_DIM, 1)
        kc = keep_col.float().view(B, 1, MAX_DIM, 1)
        iota_c = torch.arange(MAX_DIM, device=content.device, dtype=torch.float32).view(1, 1, 1, MAX_DIM)
        EcB = (pc == iota_c)
        T = EcB.float() * kc
        
        return torch.matmul(torch.matmul(Sr, content), T)

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        B, C, H, W = x.shape
        MAX_DIM = 30
        
        mask_r = torch.max(valid_mask, dim=3, keepdim=True)[0] # [B, 1, MAX_DIM, 1]
        mask_c = torch.max(valid_mask, dim=2, keepdim=True)[0] # [B, 1, 1, MAX_DIM]
        
        H_true = mask_r.sum(dim=2, keepdim=True) # [B, 1, 1, 1]
        W_true = mask_c.sum(dim=3, keepdim=True) # [B, 1, 1, 1]
        
        half_H = torch.floor(H_true / 2.0)
        half_W = torch.floor(W_true / 2.0)
        
        iota_r = torch.arange(MAX_DIM, device=x.device, dtype=torch.float32).view(1, 1, MAX_DIM, 1)
        iota_c = torch.arange(MAX_DIM, device=x.device, dtype=torch.float32).view(1, 1, 1, MAX_DIM)
        
        if self.fold_axis == 0:
            keep_row_1 = (iota_r < half_H)
            keep_row_2 = (iota_r >= H_true - half_H) & (iota_r < H_true)
            keep_col_1 = mask_c > 0.5
            keep_col_2 = mask_c > 0.5
            out_valid_mask = keep_row_1.float() * keep_col_1.float()
        else:
            keep_row_1 = mask_r > 0.5
            keep_row_2 = mask_r > 0.5
            keep_col_1 = (iota_c < half_W)
            keep_col_2 = (iota_c >= W_true - half_W) & (iota_c < W_true)
            out_valid_mask = keep_row_1.float() * keep_col_1.float()
            
        flip_r = self.flip_second and self.fold_axis == 0
        flip_c = self.flip_second and self.fold_axis == 1
        
        part1 = self._crop_compact(x, keep_row_1, keep_col_1, B, MAX_DIM)
        part2 = self._crop_compact(x, keep_row_2, keep_col_2, B, MAX_DIM, flip_r=flip_r, flip_c=flip_c)
        
        mask1 = part1[:, self.target_color:self.target_color+1, :, :]
        mask2 = part2[:, self.target_color:self.target_color+1, :, :]
        
        if self.op == 'AND':
            mask_out = mask1 * mask2
        elif self.op == 'OR':
            mask_out = torch.max(mask1, mask2)
        elif self.op == 'XOR':
            mask_out = torch.abs(mask1 - mask2)
        elif self.op == 'LEFT_NOT_RIGHT':
            mask_out = mask1 * (1.0 - mask2)
        elif self.op == 'RIGHT_NOT_LEFT':
            mask_out = mask2 * (1.0 - mask1)
            
        # Instead of replacing, we return a new array filled with zeros except for mask_out -> output_color
        out_colors = torch.zeros_like(x)
        out_colors[:, self.output_color:self.output_color+1, :, :] = mask_out
        out_colors[:, 0:1, :, :] = 1.0 - mask_out
        
        return out_colors * out_valid_mask

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
