import json
import os
import glob
from neurogolf_gemini.templates.color_map import fit_and_generate as color_map_fit
from neurogolf_gemini.templates.translation import fit_and_generate as translation_fit
from neurogolf_gemini.templates.fill_holes import fit_and_generate as fill_holes_fit
from neurogolf_gemini.templates.morphology import fit_and_generate as morphology_fit
from neurogolf_gemini.templates.ray_casting import fit_and_generate as ray_casting_fit

files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task???.json')

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
    manifest = json.load(f)

for f in sorted(files):
    task_id = os.path.basename(f).replace('.json', '')
    if 'deployed_score' in manifest.get('tasks', {}).get(task_id, {}):
        continue
        
    with open(f, 'r') as fp:
        data = json.load(fp)
        
    for name, func in [('color_map', color_map_fit),
                       ('translation', translation_fit),
                       ('fill_holes', fill_holes_fit),
                       ('morphology', morphology_fit),
                       ('ray_casting', ray_casting_fit)]:
        try:
            res = func(data, task_id)
            if res is not None:
                print(f"[{task_id}] Fits {name}!")
                break
        except Exception as e:
            # ignore exceptions in templates
            pass
