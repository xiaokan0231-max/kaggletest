import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

class Task066(nn.Module):
    def __init__(self, start_c=3, target_c=2, obs_c=8):
        super().__init__()
        self.start_c = start_c
        self.target_c = target_c
        self.obs_c = obs_c
        self.register_buffer('kernel', torch.tensor([[[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]]]))
        
    def forward(self, x):
        orig_H, orig_W = x.shape[2], x.shape[3]
        
        x_pad = F.pad(x, (0, 30 - orig_W, 0, 30 - orig_H))
        inp = torch.argmax(x_pad, dim=1) # [B, 30, 30]
        H, W = 30, 30
        
        target_mask = (inp == self.target_c).float()
        Y_grid, X_grid = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        
        target_sum = target_mask.sum(dim=[1, 2], keepdim=True)
        target_sum = target_sum + (target_sum == 0).float() * 1.0
        
        target_y_center = (target_mask * Y_grid).sum(dim=[1, 2], keepdim=True) / target_sum
        target_x_center = (target_mask * X_grid).sum(dim=[1, 2], keepdim=True) / target_sum
        
        start_mask = (inp == self.start_c).float().unsqueeze(1) # [B, 1, H, W]
        neighbors = F.conv2d(start_mask, self.kernel, padding=1).squeeze(1) # [B, H, W]
        
        tips_mask = ((inp == self.start_c) & (neighbors == 1)).float()
        
        B = inp.shape[0]
        flat_tips = tips_mask.reshape(B, -1)
        vals, indices = torch.topk(flat_tips, 2, dim=1) # [B, 2]
        
        best_score = torch.full((B,), -1000.0, device=x.device)
        best_y = torch.zeros((B,), dtype=torch.long, device=x.device)
        best_x = torch.zeros((B,), dtype=torch.long, device=x.device)
        best_dir = torch.zeros((B,), dtype=torch.long, device=x.device)
        
        pad_inp = F.pad(inp, (1, 1, 1, 1), value=self.obs_c) # [B, 32, 32]
        Y_pad, X_pad = torch.meshgrid(torch.arange(H+2, device=x.device), torch.arange(W+2, device=x.device), indexing='ij')
        Y_pad = Y_pad.unsqueeze(0) # [1, 32, 32]
        X_pad = X_pad.unsqueeze(0) # [1, 32, 32]
        
        target_x_center = target_x_center.squeeze(1).squeeze(1) # [B]
        target_y_center = target_y_center.squeeze(1).squeeze(1) # [B]
        batch_idx = torch.arange(B, device=x.device)
        
        for k in range(2):
            idx = indices[:, k] # [B]
            y = idx // W # [B]
            x = idx % W # [B]
            
            y_pad_k = y + 1
            x_pad_k = x + 1
            
            is_valid = vals[:, k] > 0.5 # [B]
            
            n_R = pad_inp[batch_idx, y_pad_k, x_pad_k+1] == self.start_c
            n_D = pad_inp[batch_idx, y_pad_k+1, x_pad_k] == self.start_c
            n_L = pad_inp[batch_idx, y_pad_k, x_pad_k-1] == self.start_c
            n_U = pad_inp[batch_idx, y_pad_k-1, x_pad_k] == self.start_c
            
            dir_k = n_L.long() * 0 + n_U.long() * 1 + n_R.long() * 2 + (~(n_L | n_U | n_R)).long() * 3
                    
            y_pad_k_expand = y_pad_k.view(B, 1, 1)
            x_pad_k_expand = x_pad_k.view(B, 1, 1)
                    
            mask0 = (Y_pad == y_pad_k_expand) & (X_pad > x_pad_k_expand) & (pad_inp == self.obs_c)
            X_obs0 = mask0.long() * X_pad + (~mask0).long() * 1000
            d0 = X_obs0.min(dim=2)[0].min(dim=1)[0] - x_pad_k - 1
            
            mask1 = (X_pad == x_pad_k_expand) & (Y_pad > y_pad_k_expand) & (pad_inp == self.obs_c)
            Y_obs1 = mask1.long() * Y_pad + (~mask1).long() * 1000
            d1 = Y_obs1.min(dim=2)[0].min(dim=1)[0] - y_pad_k - 1
            
            mask2 = (Y_pad == y_pad_k_expand) & (X_pad < x_pad_k_expand) & (pad_inp == self.obs_c)
            X_obs2 = mask2.long() * X_pad + (~mask2).long() * (-1000)
            d2 = x_pad_k - X_obs2.max(dim=2)[0].max(dim=1)[0] - 1
            
            mask3 = (X_pad == x_pad_k_expand) & (Y_pad < y_pad_k_expand) & (pad_inp == self.obs_c)
            Y_obs3 = mask3.long() * Y_pad + (~mask3).long() * (-1000)
            d3 = y_pad_k - Y_obs3.max(dim=2)[0].max(dim=1)[0] - 1
            
            dist = (dir_k == 0).long() * d0 + (dir_k == 1).long() * d1 + (dir_k == 2).long() * d2 + (dir_k == 3).long() * d3
                   
            m0 = (target_x_center > x.float()).float()
            m1 = (target_y_center > y.float()).float()
            m2 = (target_x_center < x.float()).float()
            m3 = (target_y_center < y.float()).float()
            
            moving_towards = (dir_k == 0).float() * m0 + (dir_k == 1).float() * m1 + (dir_k == 2).float() * m2 + (dir_k == 3).float() * m3
                             
            score = dist.float() * 10.0 + moving_towards
            score = is_valid.float() * score + (~is_valid).float() * (-1000.0)
            
            is_better = score > best_score
            best_score = is_better.float() * score + (~is_better).float() * best_score
            best_y = is_better.long() * y + (~is_better).long() * best_y
            best_x = is_better.long() * x + (~is_better).long() * best_x
            best_dir = is_better.long() * dir_k + (~is_better).long() * best_dir
            
        pos_y = best_y + 1 # [B]
        pos_x = best_x + 1 # [B]
        curr_dir = best_dir # [B]
        
        out_pred = pad_inp.clone()
        reached_target = torch.zeros((B,), dtype=torch.bool, device=x.device)
        
        for segment in range(4):
            y_expand = pos_y.view(B, 1, 1)
            x_expand = pos_x.view(B, 1, 1)
            dir_expand = curr_dir.view(B, 1, 1)
            
            mask0 = (Y_pad == y_expand) & (X_pad > x_expand) & ((pad_inp == self.obs_c) | (pad_inp == self.target_c))
            X_obs0 = mask0.long() * X_pad + (~mask0).long() * 1000
            d_obs0 = X_obs0.min(dim=2)[0].min(dim=1)[0] - pos_x
            
            mask1 = (X_pad == x_expand) & (Y_pad > y_expand) & ((pad_inp == self.obs_c) | (pad_inp == self.target_c))
            Y_obs1 = mask1.long() * Y_pad + (~mask1).long() * 1000
            d_obs1 = Y_obs1.min(dim=2)[0].min(dim=1)[0] - pos_y
            
            mask2 = (Y_pad == y_expand) & (X_pad < x_expand) & ((pad_inp == self.obs_c) | (pad_inp == self.target_c))
            X_obs2 = mask2.long() * X_pad + (~mask2).long() * (-1000)
            d_obs2 = pos_x - X_obs2.max(dim=2)[0].max(dim=1)[0]
            
            mask3 = (X_pad == x_expand) & (Y_pad < y_expand) & ((pad_inp == self.obs_c) | (pad_inp == self.target_c))
            Y_obs3 = mask3.long() * Y_pad + (~mask3).long() * (-1000)
            d_obs3 = pos_y - Y_obs3.max(dim=2)[0].max(dim=1)[0]
            
            step = (curr_dir == 0).long() * d_obs0 + (curr_dir == 1).long() * d_obs1 + (curr_dir == 2).long() * d_obs2 + (curr_dir == 3).long() * d_obs3
                   
            hit_y = pos_y + (curr_dir == 1).long() * step - (curr_dir == 3).long() * step
            hit_x = pos_x + (curr_dir == 0).long() * step - (curr_dir == 2).long() * step
                    
            hit_y_expand = hit_y.view(B, 1, 1)
            hit_x_expand = hit_x.view(B, 1, 1)
            hit_mask = (Y_pad == hit_y_expand) & (X_pad == hit_x_expand)
            hit_c = (pad_inp * hit_mask.long()).max(dim=2)[0].max(dim=1)[0]
            
            is_target = (hit_c == self.target_c)
            step = step - (hit_c == self.obs_c).long()
            
            step = step * (~reached_target).long()
            reached_target = reached_target | is_target
            
            step_expand = step.view(B, 1, 1)
            draw0 = (Y_pad == y_expand) & (X_pad >= x_expand) & (X_pad <= x_expand + step_expand)
            draw1 = (X_pad == x_expand) & (Y_pad >= y_expand) & (Y_pad <= y_expand + step_expand)
            draw2 = (Y_pad == y_expand) & (X_pad <= x_expand) & (X_pad >= x_expand - step_expand)
            draw3 = (X_pad == x_expand) & (Y_pad <= y_expand) & (Y_pad >= y_expand - step_expand)
            
            draw_mask = (dir_expand == 0) & draw0 | (dir_expand == 1) & draw1 | (dir_expand == 2) & draw2 | (dir_expand == 3) & draw3
                        
            out_pred = draw_mask.long() * self.start_c + (~draw_mask).long() * out_pred
            
            pos_x = pos_x + (curr_dir == 0).long() * step - (curr_dir == 2).long() * step
            pos_y = pos_y + (curr_dir == 1).long() * step - (curr_dir == 3).long() * step
                    
            target_y_c_pad = target_y_center + 1.0
            target_x_c_pad = target_x_center + 1.0
            
            next_dir_horiz = (target_y_c_pad > pos_y.float()).long() * 1 + (target_y_c_pad <= pos_y.float()).long() * 3
            next_dir_vert = (target_x_c_pad > pos_x.float()).long() * 0 + (target_x_c_pad <= pos_x.float()).long() * 2
            
            cond = (curr_dir == 0) | (curr_dir == 2)
            curr_dir = cond.long() * next_dir_horiz + (~cond).long() * next_dir_vert

        out_pred[pad_inp == self.target_c] = self.target_c
        out_final = out_pred[:, 1:-1, 1:-1] # [B, 30, 30]
        
        out_one_hot = (torch.arange(10, device=x.device).view(1, 10, 1, 1).float() == out_final.unsqueeze(1).float()).float()
        
        return out_one_hot[:, :, :orig_H, :orig_W]

if __name__ == "__main__":
    with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task066.json") as f:
        task_data = json.load(f)
    
    ex = task_data['train'][0]
    inp = np.array(ex['input'])
    
    model = Task066()
    model.eval()
    
    inp_t = torch.tensor(inp, dtype=torch.long)
    inp_onehot = F.one_hot(inp_t, num_classes=10).permute(2, 0, 1).unsqueeze(0).float()
    
    out_t = model(inp_onehot)
    print("Python Output Match:", np.array_equal(torch.argmax(out_t, dim=1).squeeze(0).numpy(), ex['output']))
    
    dummy_input = torch.zeros((1, 10, 20, 20))
    dummy_input[0, 0, :, :] = 1.0
    
    torch.onnx.export(
        model, 
        dummy_input, 
        "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/task066.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        opset_version=14
    )
    print("Exported task066.onnx")
