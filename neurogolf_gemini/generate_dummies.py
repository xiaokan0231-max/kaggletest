import torch
import torch.nn as nn
import os
import glob
import onnx

class DummyNet(nn.Module):
    def forward(self, x):
        # Always output a 1x10x1x1 tensor of color 0
        return torch.zeros(1, 10, 1, 1)

def generate_dummies():
    base_dir = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"
    working_dir = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
    
    # Also copy Codex's ONNX to working_dir
    os.system("cp /Users/kanxiao/IdeaProjects/kaggletest/neurogolf_codex/solutions/*.onnx " + working_dir)
    
    existing_onnx = set([os.path.basename(f).split(".")[0] for f in glob.glob(os.path.join(working_dir, "*.onnx"))])
    
    model = DummyNet()
    model.eval()
    
    dummy_input = torch.zeros(1, 10, 30, 30)
    
    for i in range(400):
        task_name = f"task{i:03d}"
        if task_name not in existing_onnx:
            onnx_path = os.path.join(working_dir, f"{task_name}.onnx")
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
            )

if __name__ == "__main__":
    generate_dummies()
    print("Done generating dummy ONNX models.")
