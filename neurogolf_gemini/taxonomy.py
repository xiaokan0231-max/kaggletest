import json
import os
import glob
import csv
from collections import defaultdict

def analyze_tasks(data_dir, output_csv):
    files = sorted(glob.glob(os.path.join(data_dir, "task*.json")))

    with open(output_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['task_id', 'shape_category', 'has_arc_gen', 'notes'])

        for f in files:
            task_name = os.path.basename(f)
            with open(f, 'r') as fp:
                data = json.load(fp)

            all_shapes = []
            has_arc_gen = 'arc-gen' in data

            for split in ['train', 'test', 'arc-gen']:
                if split in data:
                    for example in data[split]:
                        inp = example['input']
                        out = example['output']
                        in_s = (len(inp), len(inp[0]))
                        out_s = (len(out), len(out[0]))
                        all_shapes.append((in_s, out_s))

            # Determine category across all examples
            is_same = all(i == o for i, o in all_shapes)
            if is_same:
                cat = "SAME_SHAPE"
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

            writer.writerow([task_name, cat, has_arc_gen, ''])

if __name__ == "__main__":
    os.makedirs("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working", exist_ok=True)
    analyze_tasks("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw", "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv")
    print("Generated task_index.csv")
