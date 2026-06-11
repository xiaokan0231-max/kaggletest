"""Show input/output grids for a task. Usage: python inspect_task.py task310 [n]"""
import json
import sys
import numpy as np

RAW = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw"


def grid_str(g):
    g = np.array(g)
    return "\n".join("".join(f"{c}" if c else "." for c in row) for row in g)


def main():
    tid = sys.argv[1]
    nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    d = json.load(open(f"{RAW}/{tid}.json"))
    exs = list(d.get("train", [])) + list(d.get("test", []))
    for k, ex in enumerate(exs[:nshow]):
        inp, out = np.array(ex["input"]), np.array(ex["output"])
        print(f"--- ex{k}  in {inp.shape} -> out {out.shape} ---")
        ig = grid_str(inp).split("\n")
        og = grid_str(out).split("\n")
        wi = inp.shape[1]
        for r in range(max(len(ig), len(og))):
            li = ig[r] if r < len(ig) else ""
            lo = og[r] if r < len(og) else ""
            print(f"  {li:<{wi}}    {lo}")


if __name__ == "__main__":
    main()
