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

class TaskNet158(nn.Module):
    def forward(self, x):
        B, _, H, W = x.shape
        
        counts = torch.sum(x, dim=(2,3))
        bg_c = torch.argmax(counts, dim=1)
        
        # We need to do the c_path identification dynamically, but argmax on counts is not easy to use for dynamic indexing if we want to avoid loops.
        # Since this is for a specific task where colors are known or we can just run the loop over all 10 channels.
        
        k_dil_8 = torch.ones(1, 1, 3, 3, dtype=torch.float32, device=x.device)
        
        c_path = torch.zeros(B, dtype=torch.int64, device=x.device)
        
        # Since batch size is 1 for ONNX evaluation usually, but we want it to be robust.
        # We can just compute c_path for each color and select the one that matches the condition.
        
        # A fully parallel way to find c_path:
        is_c_path = torch.zeros(B, 10, dtype=torch.float32, device=x.device)
        
        for c in range(10):
            mask = x[:, c:c+1]
            # get one pixel: this is hard without dynamic axes.
            # but wait, can we just use max pool or something?
            # Actually, we can just use the coordinates!
            flat_mask = mask.view(B, -1)
            # find first 1
            idx = torch.argmax(flat_mask, dim=1, keepdim=True)
            seed = torch.zeros_like(flat_mask)
            seed.scatter_(1, idx, 1.0)
            seed = seed.view(B, 1, H, W) * mask
            
            for _ in range(30):
                dilated = torch.clamp(F.conv2d(seed, k_dil_8, padding=1), 0, 1)
                seed = torch.clamp(dilated * mask, 0, 1)
                
            match = (torch.sum(seed, dim=(1,2,3)) == torch.sum(mask, dim=(1,2,3))) & (torch.sum(mask, dim=(1,2,3)) > 0)
            is_c_path[:, c] = match.float()
            
        # bg_c also has one component usually (the background). We must exclude bg_c!
        # is_c_path should be false for bg_c
        for c in range(10):
            is_c_path[:, c] = is_c_path[:, c] * (bg_c != c).float()
            
        # c_path is the color that has is_c_path == 1
        # If multiple, just take argmax
        c_path_idx = torch.argmax(is_c_path, dim=1)
        
        m_path = torch.zeros(B, 1, H, W, device=x.device)
        for c in range(10):
            m_path = m_path + x[:, c:c+1] * (c_path_idx == c).view(B, 1, 1, 1).float()
            
        dilated_path = torch.clamp(F.conv2d(m_path, k_dil_8, padding=1), 0, 1)
        
        # Find cA and cB. They are the minority colors that touch m_path
        touches = torch.zeros(B, 10, dtype=torch.float32, device=x.device)
        for c in range(10):
            m_c = x[:, c:c+1]
            overlap = torch.sum(dilated_path * m_c, dim=(1,2,3)) > 0
            is_valid = (bg_c != c) & (c_path_idx != c)
            touches[:, c] = (overlap & is_valid).float()
            
        # We know there are exactly 2 touches.
        # We can extract m_A and m_B by just doing:
        # m_A + m_B = sum_{c} touches[c] * x[:, c]
        # Since we just need T_A and T_B, and they are distinct components anyway!
        # We can just sum them!
        m_AB = torch.zeros(B, 1, H, W, device=x.device)
        for c in range(10):
            m_AB = m_AB + x[:, c:c+1] * touches[:, c].view(B, 1, 1, 1)
            
        # T_A and T_B can be extracted as a single mask T_AB!
        # Wait, if we use T_AB, it contains BOTH T_A and T_B.
        # When we match, we can just require T_AB to match m_AB!
        # Is that mathematically equivalent?
        # Yes! T_AB_s will contain BOTH A and B endpoints at the correct relative distance!
        # So we don't need to match them separately!
        # corr_AB = conv2d(m_AB, T_AB_s)
        # valid_shift = (corr_AB >= sum(T_AB_s))
        # This is EVEN BETTER and FASTER!
        
        T_AB_start = torch.clamp(dilated_path * m_AB, 0, 1)
        T_AB = T_AB_start
        for _ in range(10):
            dilated_AB = torch.clamp(F.conv2d(T_AB, k_dil_8, padding=1), 0, 1)
            T_AB = torch.clamp(dilated_AB * m_AB, 0, 1)
            
        T_path = m_path
        final_path = m_path.clone()
        
        for s in range(1, 6):
            T_AB_s = F.interpolate(T_AB, scale_factor=s, mode='nearest')
            T_path_s = F.interpolate(T_path, scale_factor=s, mode='nearest')
            
            sum_AB = T_AB_s.sum()
            has_AB = (sum_AB > 0.5).view(1, 1, 1, 1)
                
            for k_t in range(8):
                T_AB_k = transform(T_AB_s, k_t)
                T_path_k = transform(T_path_s, k_t)
                
                kH, kW = T_AB_k.shape[2:]
                
                m_AB_padded = F.pad(m_AB, (kW, kW, kH, kH))
                
                # groups=B simulation
                m_AB_padded_g = m_AB_padded.view(1, B, H+2*kH, W+2*kW)
                T_AB_k_g = T_AB_k.view(B, 1, kH, kW)
                
                corr_AB = F.conv2d(m_AB_padded_g, T_AB_k_g, groups=B).view(B, 1, H+kH+1, W+kW+1)
                
                valid_shift = (corr_AB >= sum_AB - 0.5) & has_AB
                
                kernel = torch.flip(valid_shift.float(), [2, 3]).view(B, 1, H+kH+1, W+kW+1)
                
                T_path_k_padded = F.pad(T_path_k, (W, W, H, H)).view(1, B, kH+2*H, kW+2*W)
                shifted_path = F.conv2d(T_path_k_padded, kernel, groups=B).view(B, 1, H, W)
                
                final_path = torch.clamp(final_path + shifted_path, 0, 1)
                
        out = x.clone()
        mask_bg = torch.zeros(B, 1, H, W, device=x.device)
        for c in range(10):
            mask_bg = mask_bg + x[:, c:c+1] * (bg_c == c).view(B, 1, 1, 1).float()
            
        path_add = final_path * mask_bg
        
        # Need to dynamically update the output channels
        for c in range(10):
            # If c is path color, add path_add
            # If c is bg color, subtract path_add
            is_path = (c_path_idx == c).view(B, 1, 1, 1).float()
            is_bg = (bg_c == c).view(B, 1, 1, 1).float()
            
            out[:, c:c+1] = torch.clamp(out[:, c:c+1] + path_add * is_path, 0, 1)
            out[:, c:c+1] = torch.clamp(out[:, c:c+1] - path_add * is_bg, 0, 1)
            
        return out

def export_onnx():
    net = TaskNet158()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task158.onnx"
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
