import json, glob, csv, os
import numpy as np

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
    manifest = json.load(f)
    
shrink_tasks = set()
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row[1].startswith('SHRINK'):
            shrink_tasks.add(row[0].replace('.json', ''))
            
solved = 0
for f in sorted(glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task???.json')):
    task_id = os.path.basename(f).replace('.json', '')
    if task_id not in shrink_tasks: continue
    if 'deployed_score' in manifest.get('tasks', {}).get(task_id, {}): continue
    
    with open(f, 'r') as fp:
        data = json.load(fp)
        
    train_data = data['train']
    
    # Try bbox + cmap
    for target_color in list(range(10)) + [-1]:
        match = True
        cmap = {}
        for ex in train_data:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            if target_color == -1: mask = (inp != 0)
            else: mask = (inp == target_color)
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows):
                match = False; break
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            crop = inp[r_min:r_max+1, c_min:c_max+1]
            if crop.shape != out.shape:
                match = False; break
                
            # Build cmap
            for c_in in range(10):
                c_out_vals = out[crop == c_in]
                if len(c_out_vals) > 0:
                    uniq = np.unique(c_out_vals)
                    if len(uniq) > 1:
                        match = False; break
                    if c_in in cmap and cmap[c_in] != uniq[0]:
                        match = False; break
                    cmap[c_in] = uniq[0]
            if not match: break
            
        if match:
            print(f"{task_id} solved by bbox + cmap! Target color: {target_color}, Cmap: {cmap}")
            solved += 1
            break
            
print(f"Total solved: {solved}")
