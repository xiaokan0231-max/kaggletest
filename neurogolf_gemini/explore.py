import json
import os
import sys

def print_task_shapes(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Task: {os.path.basename(filepath)}")
    for split in ['train', 'test', 'arc-gen']:
        if split in data:
            print(f"\n--- {split} ---")
            for i, example in enumerate(data[split]):
                if i >= 3 and split == 'arc-gen': continue
                inp = example['input']
                out = example['output']
                print(f"Example {i}: Input {len(inp)}x{len(inp[0])} -> Output {len(out)}x{len(out[0])}")
                if i == 0:
                    print("Input:")
                    for row in inp: print(row)
                    print("Output:")
                    for row in out: print(row)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print_task_shapes(sys.argv[1])
    else:
        print("Please provide a path to a json file.")
