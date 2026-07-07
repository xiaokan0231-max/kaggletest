import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
import json

def load_task(filepath="neurogolf/data/raw/task066.json"):
    with open(filepath) as f:
        data = json.load(f)
    return data['train']

def build_model():
    # Build a simple ONNX model to test logic.
    # For now, let's just do a simple dummy model to ensure the pipeline works.
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10, 30, 30])
    
    # Just Identity for now
    node = helper.make_node('Identity', ['input'], ['output'])
    
    graph = helper.make_graph([node], 'test_graph', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model

if __name__ == "__main__":
    train_data = load_task()
    model = build_model()
    ort_session = ort.InferenceSession(model.SerializeToString())
    
    for ex in train_data:
        inp = np.array(ex['input'], dtype=np.float32)
        
        # Convert to one-hot 1x10x30x30
        h, w = inp.shape
        one_hot = np.zeros((1, 10, 30, 30), dtype=np.float32)
        for r in range(h):
            for c in range(w):
                one_hot[0, int(inp[r, c]), r, c] = 1.0
                
        out = ort_session.run(None, {'input': one_hot})[0]
        print(f"Output shape: {out.shape}")
        break
