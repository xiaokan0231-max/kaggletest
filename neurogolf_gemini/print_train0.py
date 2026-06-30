import json
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task018.json') as f:
    data = json.load(f)

ex = data['train'][0]
print("True output unique:", set(c for r in ex['output'] for c in r))
