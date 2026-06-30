import sys
import onnx
import numpy as np
import onnxruntime
sys.path.insert(0, '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils
neurogolf_utils._NEUROGOLF_DIR = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/"
from neurogolf_utils import load_examples, convert_to_numpy, verify_network

onnx_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task013.onnx"
session = onnxruntime.InferenceSession(onnx_path)
examples = load_examples(13)

model = onnx.load(onnx_path)
verify_network(model, 13, examples)
