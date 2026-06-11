import json
import numpy as np

def solve(grid):
    a = np.array(grid, dtype=int)
    out = np.pad(a, 1, mode='edge')
    out[0, 0] = 0
    out[0, -1] = 0
    out[-1, 0] = 0
    out[-1, -1] = 0
    return out.tolist()

if __name__ == '__main__':
    d = json.load(open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task114.json'))
    for split in ['train', 'test', 'arc-gen']:
        ok = 0
        fails = []
        for i, ex in enumerate(d[split]):
            got = solve(ex['input'])
            if got == ex['output']:
                ok += 1
            else:
                fails.append(i)
        print(split, f'{ok}/{len(d[split])}', 'FAIL:' if fails else '', fails[:5])
    # also report size ranges
    hs = [len(e['input']) for e in d['arc-gen']]
    ws = [len(e['input'][0]) for e in d['arc-gen']]
    print('arc-gen input h range', min(hs), max(hs), 'w range', min(ws), max(ws))
