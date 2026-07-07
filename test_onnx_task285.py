import json
import numpy as np
import onnxruntime as ort

with open('neurogolf/data/raw/task285.json') as f:
    data = json.load(f)

# Load the ONNX model
ort_session = ort.InferenceSession("task285.onnx", providers=["CPUExecutionProvider"])
input_name = ort_session.get_inputs()[0].name

print("Verifying task285.onnx...")
all_match = True

for i, ex in enumerate(data['train']):
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    H, W = inp.shape
    x_ts = np.zeros((1, 10, H, W), dtype=np.float32)
    for c in range(10):
        x_ts[0, c] = (inp == c).astype(np.float32)
        
    # Run the model
    ort_outs = ort_session.run(None, {input_name: x_ts})
    out_pred = ort_outs[0]
    
    out_pred_label = np.argmax(out_pred[0], axis=0)
    
    match = np.array_equal(out_pred_label, out)
    print(f"Train {i}: {'✅ Match' if match else '❌ Mismatch'}")
    if not match:
        all_match = False
        diff = (out_pred_label != out)
        for r, c in zip(*np.where(diff)):
            print(f"  Diff at ({r}, {c}): Pred {out_pred_label[r, c]}, True {out[r, c]}")

if all_match:
    print("\n✅ All training cases passed via ONNX Runtime!")
else:
    print("\n❌ Verification failed on ONNX Runtime.")
