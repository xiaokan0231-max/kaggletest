import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class Task017Net(nn.Module):
    def get_tiled_h(self, x, h):
        num_blocks = (30 + h - 1) // h
        pad_len = num_blocks * h - 30
        x_pad = F.pad(x, (0, 0, 0, pad_len))
        x_reshaped = x_pad.view(1, 9, num_blocks, h, 30)
        folded = torch.max(x_reshaped, dim=2, keepdim=True)[0]
        expanded = folded.expand(1, 9, num_blocks, h, 30)
        tiled = expanded.reshape(1, 9, num_blocks * h, 30)
        return tiled[:, :, :30, :]

    def get_tiled_w(self, x, w):
        num_blocks = (30 + w - 1) // w
        pad_len = num_blocks * w - 30
        x_pad = F.pad(x, (0, pad_len, 0, 0))
        x_reshaped = x_pad.view(1, 9, 30, num_blocks, w)
        folded = torch.max(x_reshaped, dim=3, keepdim=True)[0]
        expanded = folded.expand(1, 9, 30, num_blocks, w)
        tiled = expanded.reshape(1, 9, 30, num_blocks * w)
        return tiled[:, :, :, :30]

    def forward(self, x):
        valid_mask = torch.max(x, dim=1, keepdim=True)[0]
        x_colors = x[:, 1:, :, :] # (1, 9, 30, 30)
        
        costs_h = []
        tiled_hs = []
        for h in range(1, 22):
            t_h = self.get_tiled_h(x_colors, h)
            conflict = torch.sum((torch.sum(t_h, dim=1) > 1).float())
            cost = conflict * 10000.0 + h
            tiled_hs.append(t_h)
            costs_h.append(cost)
        
        best_h_idx = torch.argmin(torch.stack(costs_h))
        tiled_hs_tensor = torch.stack(tiled_hs)
        best_t_h = torch.index_select(tiled_hs_tensor, 0, best_h_idx.view(1)).squeeze(0)
        
        costs_w = []
        tiled_ws = []
        for w in range(1, 22):
            t_w = self.get_tiled_w(x_colors, w)
            conflict = torch.sum((torch.sum(t_w, dim=1) > 1).float())
            cost = conflict * 10000.0 + w
            tiled_ws.append(t_w)
            costs_w.append(cost)
            
        best_w_idx = torch.argmin(torch.stack(costs_w))
        tiled_ws_tensor = torch.stack(tiled_ws)
        best_t_w = torch.index_select(tiled_ws_tensor, 0, best_w_idx.view(1)).squeeze(0)
        
        combined = torch.max(best_t_h, best_t_w)
        
        out = torch.zeros_like(x)
        out[:, 1:, :, :] = combined
        max_color = torch.max(combined, dim=1, keepdim=True)[0]
        out[:, 0:1, :, :] = 1.0 - torch.clamp(max_color, 0.0, 1.0)
        
        return out * valid_mask

def export_onnx():
    net = Task017Net()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task017.onnx"
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Exported to {onnx_path}")

if __name__ == "__main__":
    export_onnx()
