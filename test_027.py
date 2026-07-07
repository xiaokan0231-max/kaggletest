import json, numpy as np
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task027.json') as f: d = json.load(f)
for ex in d['train']:
  inp = np.array(ex['input'])
  out = np.array(ex['output'])
  print('Input:')
  for r in inp: print(''.join(str(x) if x!=0 else '.' for x in r))
  print('Output:')
  for r in out: print(''.join(str(x) if x!=0 else '.' for x in r))
