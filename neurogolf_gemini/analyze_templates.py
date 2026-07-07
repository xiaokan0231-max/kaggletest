import json
import os
import glob
import numpy as np

def analyze_tasks():
    files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task*.json')
    same_shape_tasks = []
    
    is_color_map = 0
    
    for f in sorted(files):
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        is_same_shape = True
        # Note: the json contains dicts where values might be list of dicts.
        if 'train' not in data:
            # maybe list of dicts?
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
            
        same_shape_tasks.append(os.path.basename(f))
        
        # Check if pure color mapping
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
            is_color_map += 1

    print(f"Total SAME_SHAPE tasks (by shape check): {len(same_shape_tasks)}")
    print(f"Tasks that are pure color mapping: {is_color_map}")

if __name__ == '__main__':
    analyze_tasks()
