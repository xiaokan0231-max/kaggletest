import os
import sys
import onnx

# Add neurogolf_utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neurogolf', 'data', 'raw', 'neurogolf_utils'))
import neurogolf_utils
neurogolf_utils._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
from neurogolf_utils import load_examples, verify_network

def test_075():
    task_json_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task075.json"
    onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task075.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} does not exist.")
        return

    print("Verifying task075...")
    
    examples = load_examples(75)
    model = onnx.load(onnx_path)
    
    try:
        verify_network(model, 75, examples)
        print("\nSuccess! ONNX model solved the task.")
    except Exception as e:
        print(f"\nFailed! Exception: {e}")

if __name__ == "__main__":
    test_075()
