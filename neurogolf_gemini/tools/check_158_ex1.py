import json
import numpy as np
import scipy.ndimage as ndimage

with open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task158.json") as f:
    task_data = json.load(f)

ex = task_data['train'][1]
inp = np.array(ex['input'])

conn_mask = (inp == 1)
p1_mask = (inp == 6)
p2_mask = (inp == 2)

s = ndimage.generate_binary_structure(2, 2)
conn_dilated = ndimage.binary_dilation(conn_mask, structure=s)

p1_temp = p1_mask & conn_dilated
p2_temp = p2_mask & conn_dilated

print("Template:")
for r, c in zip(*np.where(p1_temp)): print(f"  6 at ({r}, {c})")
for r, c in zip(*np.where(p2_temp)): print(f"  2 at ({r}, {c})")

print("Targets:")
labeled, n = ndimage.label(p1_mask, structure=s)
for i in range(1, n+1):
    mask = (labeled == i)
    if not np.any(mask & conn_dilated):
        r, c = np.where(mask)
        print(f"  Target 6 at ({r.min()}:{r.max()+1}, {c.min()}:{c.max()+1}) size {len(r)}")
        
labeled, n = ndimage.label(p2_mask, structure=s)
for i in range(1, n+1):
    mask = (labeled == i)
    if not np.any(mask & conn_dilated):
        r, c = np.where(mask)
        print(f"  Target 2 at ({r.min()}:{r.max()+1}, {c.min()}:{c.max()+1}) size {len(r)}")
