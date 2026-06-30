import neurogolf_utils
import json
import traceback

try:
    neurogolf_utils.verify_network('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task006.onnx', 'task006')
except Exception as e:
    traceback.print_exc()
