import json, numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    d = json.load(f)

for i in [1, 2]:
    ex = d['train'][i]
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    
    box_color = 2
    obj_color = [c for c in np.unique(inp) if c != 0 and c != box_color][0]
    
    mask_obj = (inp == obj_color)
    r_obj, c_obj = np.where(mask_obj)
    o_rmin, o_rmax = np.min(r_obj), np.max(r_obj)
    o_cmin, o_cmax = np.min(c_obj), np.max(c_obj)
    obj_crop = inp[o_rmin:o_rmax+1, o_cmin:o_cmax+1]
    
    print(f'--- Example {i} ---')
    print('Object Crop:')
    print(obj_crop)
    print('Output:')
    print(out)
