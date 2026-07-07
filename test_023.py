import json, numpy as np
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task023.json') as f:
    d = json.load(f)
for ex in d['train']:
    inp = np.array(ex['input'])
    out = np.array(ex['output'])
    diff = inp != out
    # show what 5 turns into
    idx5 = (inp == 5)
    print("5s become:", np.unique(out[idx5]))
