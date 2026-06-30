import json
import sys

def view_task(task_id):
    path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/{task_id}.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {task_id}: {e}")
        return
        
    print(f"--- {task_id} ---")
    for i, ex in enumerate(data['train']):
        print(f"Train {i}:")
        inp, out = ex['input'], ex['output']
        print(f"  Shape IN: {len(inp)}x{len(inp[0])}  OUT: {len(out)}x{len(out[0])}")
        
        # We can just print them side by side
        for r1, r2 in zip(inp, out):
            print("  " + "".join(str(x) if x!=0 else '.' for x in r1) + " | " + "".join(str(x) if x!=0 else '.' for x in r2))
        print()

if __name__ == "__main__":
    for t in sys.argv[1:]:
        view_task(t)
