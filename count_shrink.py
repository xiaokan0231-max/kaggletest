import json, csv

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
    manifest = json.load(f)

solved = set(manifest.get('tasks', {}).keys())

shrink_total = 0
shrink_solved = 0

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row[1].startswith('SHRINK'):
            task_id = row[0].replace('.json', '')
            shrink_total += 1
            if task_id in solved:
                shrink_solved += 1

print(f"SHRINK tasks solved: {shrink_solved} / {shrink_total}")
