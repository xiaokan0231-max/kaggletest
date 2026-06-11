"""task031 numpy reference: crop grid to bounding box of nonzero cells."""
import numpy as np

def solve(grid):
    a = np.array(grid)
    rows = np.where(a.any(axis=1))[0]
    cols = np.where(a.any(axis=0))[0]
    return a[rows.min():rows.max()+1, cols.min():cols.max()+1].tolist()

if __name__ == "__main__":
    import json
    d = json.load(open("/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task031.json"))
    for split in ("train", "test", "arc-gen"):
        ok = 0
        for i, ex in enumerate(d[split]):
            got = solve(ex["input"])
            if got == ex["output"]:
                ok += 1
            else:
                print("FAIL", split, i)
                print("IN:");  [print(''.join(map(str,r))) for r in ex["input"]]
                print("EXPECT:"); [print(''.join(map(str,r))) for r in ex["output"]]
                print("GOT:"); [print(''.join(map(str,r))) for r in got]
        print(split, "%d/%d" % (ok, len(d[split])))
