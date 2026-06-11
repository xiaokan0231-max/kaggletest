import json
import numpy as np

def solve(grid):
    """task019: tile input 2x2; every background cell diagonally adjacent
    (4 diagonal neighbors) to a colored cell in the tiled grid becomes 8."""
    a = np.array(grid, dtype=int)
    t = np.tile(a, (2, 2))
    H, W = t.shape
    colored = (t != 0)
    # diagonal-only dilation
    pad = np.zeros((H + 2, W + 2), dtype=bool)
    pad[1:-1, 1:-1] = colored
    diag = pad[0:-2, 0:-2] | pad[0:-2, 2:] | pad[2:, 0:-2] | pad[2:, 2:]
    out = t.copy()
    out[(t == 0) & diag] = 8
    return out.tolist()

if __name__ == "__main__":
    d = json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task019.json'))
    for split in ['train', 'test', 'arc-gen']:
        ok = 0
        fails = []
        for i, ex in enumerate(d[split]):
            got = solve(ex['input'])
            if got == ex['output']:
                ok += 1
            else:
                fails.append(i)
        print(f"{split}: {ok}/{len(d[split])} pass", "FAILS:" if fails else "", fails[:10])
    for split in ['train', 'test', 'arc-gen']:
        for i, ex in enumerate(d[split]):
            if solve(ex['input']) != ex['output']:
                print('--- first failure', split, i)
                print('input:')
                for row in ex['input']: print(' '.join(map(str, row)))
                print('expected:')
                for row in ex['output']: print(' '.join(map(str, row)))
                print('got:')
                for row in solve(ex['input']): print(' '.join(map(str, row)))
                raise SystemExit
