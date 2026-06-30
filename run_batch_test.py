import json
import os
import glob
from neurogolf_gemini.templates.color_map import fit_and_generate as color_map_fit
from neurogolf_gemini.templates.translation import fit_and_generate as translation_fit
from neurogolf_gemini.templates.fill_holes import fit_and_generate as fill_holes_fit
from neurogolf_gemini.templates.morphology import fit_and_generate as morphology_fit
from neurogolf_gemini.templates.ray_casting import fit_and_generate as ray_casting_fit

files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task025.json')
for f in files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    print("Testing fill_holes on", os.path.basename(f))
    res = fill_holes_fit(data, 'task025')
    print("fill_holes done.")
    print("Testing morphology on", os.path.basename(f))
    res = morphology_fit(data, 'task025')
    print("morphology done.")
