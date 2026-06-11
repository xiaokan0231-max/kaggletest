import json
import os
import glob
import numpy as np

def run():
    files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task*.json')
    
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    for f in sorted(files):
        task_id = os.path.basename(f).replace('.json', '')
        # Skip if already deployed
        task_info = manifest.get('tasks', {}).get(task_id, {})
        if 'deployed_score' in task_info and not task_info.get('is_dummy', True):
            # Actually, let's just count for now
            pass
            
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        is_same_shape = True
        if 'train' not in data:
            if isinstance(data, list):
                train_data = [d for d in data if 'output' in d]
            else:
                continue
        else:
            train_data = data['train']
            
        for ex in train_data:
            if np.array(ex['input']).shape != np.array(ex['output']).shape:
                is_same_shape = False
                break
                
        if not is_same_shape:
            continue
            
        mapping = {}
        valid_mapping = True
        for ex in train_data:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    ci = inp[i,j]
                    co = out[i,j]
                    if ci not in mapping:
                        mapping[ci] = co
                    elif mapping[ci] != co:
                        valid_mapping = False
                        break
                if not valid_mapping: break
            if not valid_mapping: break
            
        if valid_mapping:
            print(f"Task {task_id} is a pure color mapping: {mapping}")

if __name__ == '__main__':
    run()
