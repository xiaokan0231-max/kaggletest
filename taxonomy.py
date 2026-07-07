import os
import json
import csv
import glob
import re
from pathlib import Path

def analyze_task(task_data):
    """Analyze a single ARC-AGI task dict and return taxonomy features."""
    train_pairs = task_data.get('train', [])
    
    if not train_pairs:
        return {'shape_change': 'UNKNOWN', 'color_conservation': 'UNKNOWN'}
    
    shape_changes = set()
    color_conserved = True
    
    for pair in train_pairs:
        inp = pair.get('input', [])
        out = pair.get('output', [])
        
        in_h, in_w = len(inp), len(inp[0]) if inp else 0
        out_h, out_w = len(out), len(out[0]) if out else 0
        
        # Analyze shape
        if in_h == out_h and in_w == out_w:
            shape_changes.add('SAME')
        elif out_h <= in_h and out_w <= in_w:
            shape_changes.add('SHRINK')
        elif out_h >= in_h and out_w >= in_w:
            shape_changes.add('EXPAND')
        else:
            shape_changes.add('DYNAMIC')
            
        # Analyze colors
        in_colors = set(val for row in inp for val in row)
        out_colors = set(val for row in out for val in row)
        
        # If output contains colors not present in input, it's not conserved
        if not out_colors.issubset(in_colors):
            color_conserved = False
            
    # Determine overall shape change label
    if len(shape_changes) == 1:
        overall_shape = list(shape_changes)[0]
    elif 'DYNAMIC' in shape_changes or ('SHRINK' in shape_changes and 'EXPAND' in shape_changes):
        overall_shape = 'DYNAMIC'
    else:
        overall_shape = 'MIXED'
        
    return {
        'shape_change': overall_shape,
        'color_conservation': 'CONSERVED' if color_conserved else 'NOT_CONSERVED'
    }

def main():
    base_dir = Path(__file__).resolve().parent
    raw_data_dir = base_dir / 'neurogolf' / 'data' / 'raw'
    working_dir = base_dir / 'neurogolf' / 'data' / 'working'
    
    working_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = working_dir / 'task_index.csv'
    
    task_files = glob.glob(str(raw_data_dir / 'task[0-9][0-9][0-9].json'))
    task_files.sort()
    
    results = []
    for filepath in task_files:
        filename = os.path.basename(filepath)
        task_id_match = re.search(r'task(\d{3})\.json', filename)
        if not task_id_match:
            continue
            
        task_id = task_id_match.group(1)
        
        try:
            with open(filepath, 'r') as f:
                task_data = json.load(f)
                
            features = analyze_task(task_data)
            features['task_id'] = task_id
            results.append(features)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Write to CSV
    if results:
        fieldnames = ['task_id', 'shape_change', 'color_conservation']
        with open(out_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Successfully generated taxonomy index at {out_csv_path} with {len(results)} tasks.")
    else:
        print("No tasks processed.")

if __name__ == '__main__':
    main()
