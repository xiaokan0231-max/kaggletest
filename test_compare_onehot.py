import onnxruntime as ort
import numpy as np
import json
import sys
sys.path.append('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json') as f:
    data = json.load(f)
    
ex = data['train'][0]
benchmark = neurogolf_utils.convert_to_numpy(ex)

sess = ort.InferenceSession('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task006.onnx')
user_output = neurogolf_utils.run_network(sess, benchmark['input'])

print("Arrays equal?", np.array_equal(user_output, benchmark['output']))
diff = np.where(user_output != benchmark['output'])
print("Differences at:")
for d in zip(*diff):
    print(f"Index: {d}, user={user_output[d]}, exp={benchmark['output'][d]}")
