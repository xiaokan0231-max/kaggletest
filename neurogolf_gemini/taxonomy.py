import json
import os
import glob
import csv

def analyze_tasks(data_dir, output_csv):
    files = sorted(glob.glob(os.path.join(data_dir, "task*.json")))
    
    with open(output_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['task_id', 'rule_family', 'shape_signature', 'color_delta', 'train_count', 'test_count', 'arc_gen_count'])
        
        for f in files:
            task_name = os.path.basename(f)
            with open(f, 'r') as fp:
                data = json.load(fp)
            
            all_shapes = []
            train_c, test_c, arc_c = 0, 0, 0
            all_in_colors = set()
            all_out_colors = set()
            
            for split in ['train', 'test', 'arc-gen']:
                if split in data:
                    for example in data[split]:
                        if split == 'train': train_c += 1
                        elif split == 'test': test_c += 1
                        elif split == 'arc-gen': arc_c += 1
                        
                        inp = example['input']
                        out = example['output']
                        in_s = (len(inp), len(inp[0]))
                        out_s = (len(out), len(out[0]))
                        all_shapes.append((in_s, out_s))
                        
                        for row in inp:
                            for c in row: all_in_colors.add(c)
                        for row in out:
                            for c in row: all_out_colors.add(c)
            
            added_colors = all_out_colors - all_in_colors
            removed_colors = all_in_colors - all_out_colors
            
            cdelta = ""
            if added_colors:
                cdelta += f"+{sorted(list(added_colors))} "
            if removed_colors:
                cdelta += f"-{sorted(list(removed_colors))}"
            cdelta = cdelta.strip()
            if not cdelta:
                cdelta = "SAME"
                
            is_same = all(i == o for i, o in all_shapes)
            if is_same:
                cat = "SAME_SHAPE"
                # Check if all examples have exactly the same fixed shape, e.g. 10x10 -> 10x10
                first_in = all_shapes[0][0]
                if all(i == first_in for i, o in all_shapes):
                    shape_sig = f"{first_in[0]}x{first_in[1]}->{first_in[0]}x{first_in[1]}"
                else:
                    shape_sig = "SAME(VARIABLE)"
            else:
                is_expand = all(i[0]*i[1] < o[0]*o[1] for i, o in all_shapes)
                is_shrink = all(i[0]*i[1] > o[0]*o[1] for i, o in all_shapes)
                if is_expand:
                    is_mult = all(o[0]%i[0] == 0 and o[1]%i[1] == 0 for i, o in all_shapes)
                    cat = "EXPAND_MULTIPLIER" if is_mult else "EXPAND"
                elif is_shrink:
                    cat = "SHRINK"
                else:
                    cat = "MIXED"
                shape_sig = "VARIABLE"
                    
            writer.writerow([task_name, cat, shape_sig, cdelta, train_c, test_c, arc_c])

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    working_dir = os.path.join(base_dir, 'neurogolf', 'data', 'working')
    os.makedirs(working_dir, exist_ok=True)
    raw_dir = os.path.join(base_dir, 'neurogolf', 'data', 'raw')
    csv_path = os.path.join(working_dir, 'task_index.csv')
    analyze_tasks(raw_dir, csv_path)
    print(f"Generated {csv_path}")
