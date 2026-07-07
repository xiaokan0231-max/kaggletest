import json
from templates.shrink_fold import fit_and_generate
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task006.json') as f:
    data = json.load(f)
code = fit_and_generate(data, 'task006')
if code:
    with open('tmp_solve.py', 'w') as out:
        out.write(code)
    print("Generated tmp_solve.py")
else:
    print("Could not fit task006")
