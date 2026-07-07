import json
from templates.shrink_extract import fit_and_generate
with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task159.json') as f:
    data = json.load(f)
code = fit_and_generate(data, 'task159')
if code:
    print("Match!")
else:
    print("No Match!")
