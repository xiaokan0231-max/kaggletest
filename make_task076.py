import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import os

def transform(x, k):
    if k >= 4:
        x = torch.flip(x, [3])
        k = k - 4
    if k == 1:
        x = torch.rot90(x, 1, [2, 3])
    elif k == 2:
        x = torch.rot90(x, 2, [2, 3])
    elif k == 3:
        x = torch.rot90(x, 3, [2, 3])
    return x

class TaskNet076(nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        
        m1 = x[:, 1:2]
        m2 = x[:, 2:3]
        m3 = x[:, 3:4]
        m4 = x[:, 4:5]
        
        m13 = torch.clamp(m1 + m3, 0, 1)
        k = torch.tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=torch.float32, device=x.device)
        dilated_m13 = torch.clamp(F.conv2d(m13, k, padding=1), 0, 1)
        
        adj_4 = torch.clamp(dilated_m13 * m4, 0, 1)
        
        T4 = adj_4
        for _ in range(30):
            dilated_T4 = torch.clamp(F.conv2d(T4, k, padding=1), 0, 1)
            T4 = torch.clamp(dilated_T4 * m4, 0, 1)
            
        dilated_T4 = torch.clamp(F.conv2d(T4, k, padding=1), 0, 1)
        T2 = torch.clamp(dilated_T4 * m2, 0, 1)
        T1 = m1
        T3 = m3
        
        final_out1 = torch.zeros_like(m1)
        final_out3 = torch.zeros_like(m3)
        
        S2 = m2
        m4_inv = 1.0 - m4
        
        for k_t in range(8):
            T4_k = transform(T4, k_t)
            T2_k = transform(T2, k_t)
            T1_k = transform(T1, k_t)
            T3_k = transform(T3, k_t)
            
            # Cross-correlation to find shifts
            # We use group convolution to simulate batched cross-correlation
            S2_g = S2.view(1, B, H, W)
            T2_k_g = T2_k.view(B, 1, H, W)
            corr = F.conv2d(S2_g, T2_k_g, padding=(H-1, W-1), groups=B).view(B, 1, 2*H-1, 2*W-1)
            
            m4_inv_g = m4_inv.view(1, B, H, W)
            T4_k_g = T4_k.view(B, 1, H, W)
            mismatch = F.conv2d(m4_inv_g, T4_k_g, padding=(H-1, W-1), groups=B).view(B, 1, 2*H-1, 2*W-1)
            
            valid_corr = corr * (mismatch < 0.5).float()
            
            kernel = torch.flip(valid_corr, [2, 3]).view(B, 1, 2*H-1, 2*W-1)
            
            # To apply the kernel to T1_k and T3_k for each batch element independently
            # We pad T1_k and T3_k
            T1_k_padded = F.pad(T1_k, (W-1, W-1, H-1, H-1)).view(1, B, 3*H-2, 3*W-2)
            T3_k_padded = F.pad(T3_k, (W-1, W-1, H-1, H-1)).view(1, B, 3*H-2, 3*W-2)
            
            shifted_T1 = F.conv2d(T1_k_padded, kernel, groups=B).view(B, 1, H, W)
            shifted_T3 = F.conv2d(T3_k_padded, kernel, groups=B).view(B, 1, H, W)
            
            final_out1 = torch.clamp(final_out1 + shifted_T1, 0, 1)
            final_out3 = torch.clamp(final_out3 + shifted_T3, 0, 1)
            
        out = x.clone()
        
        # Overwrite 0s
        mask_0 = x[:, 0:1]
        
        out1_add = final_out1 * mask_0
        out3_add = final_out3 * mask_0
        
        out[:, 1:2] = torch.clamp(out[:, 1:2] + out1_add, 0, 1)
        out[:, 3:4] = torch.clamp(out[:, 3:4] + out3_add, 0, 1)
        
        out[:, 0:1] = torch.clamp(out[:, 0:1] - out1_add - out3_add, 0, 1)
        
        return out

def export_onnx():
    net = TaskNet076()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task076.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=14
    )
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
