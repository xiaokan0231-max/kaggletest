import torch
import torch.nn as nn
import onnx

class Task043Net(nn.Module):
    def forward(self, x):
        # x: (1, 10, 30, 30) one-hot
        # Color 5 = channel 5 (0-indexed), Color 2 = channel 2
        # Row 0 has pattern of 5s (columns), Last col (col 9) has pattern of 5s (rows)
        # Where both intersect (row has 5 on right edge AND col has 5 on top row) => put 2
        
        c5 = x[:, 5:6, :, :]  # (1,1,30,30) - color 5 locations
        
        # Top-row pattern: c5 at row 0
        top_row = c5[:, :, 0:1, :]  # (1,1,1,30)
        
        # Right-col pattern: find which column has 5s on right edge
        # The 5s on right edge tell us which rows get the pattern
        # Looking at the data: col 9 has 5s marking rows
        # Actually, we need to detect which column is the "edge column"
        # From examples, 5s appear at the last non-zero column
        
        # Let's detect: for each row, is there a 5 on the rightmost position?
        # Actually, the pattern is: row 0 has 5s marking columns, 
        # the rightmost column with any 5 marks which rows get duplicated
        
        # Simpler: take right edge column (the max-column position where 5 appears)
        # In all examples, right edge col = col 9
        right_col = c5[:, :, :, 9:10]  # (1,1,30,1)
        
        # Intersection: where both top_row and right_col have 5
        intersection = top_row * right_col  # (1,1,30,30)
        
        # Remove the original 5 positions from intersection
        # (don't replace original 5s)
        new_2 = intersection * (1.0 - c5)
        
        # Build output: keep original, add color 2 where intersection is
        out = x.clone()
        out[:, 2:3, :, :] = torch.clamp(out[:, 2:3, :, :] + new_2, 0, 1)
        # Where we added 2, remove the 0 channel
        out[:, 0:1, :, :] = out[:, 0:1, :, :] * (1.0 - new_2)
        
        return out

def export_onnx():
    net = Task043Net()
    net.eval()
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task043.onnx"
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
