import json
import glob
import os

def grid_to_md(grid):
    colors = {
        0: '⬛', 1: '🟦', 2: '🟥', 3: '🟩',
        4: '🟨', 5: '⬜', 6: '🟪', 7: '🟧',
        8: '🩵', 9: '🟫'
    }
    md = ''
    for row in grid:
        for val in row:
            md += colors.get(val, '❓')
        md += '<br>'
    return md

files = sorted(glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task00*.json'))
md = '# Early Tasks Visualization\n\n'

for f in files:
    task_name = os.path.basename(f)
    md += f'## {task_name}\n\n'
    with open(f) as fp:
        data = json.load(fp)
    
    md += '| Split | Input | Output |\n'
    md += '|---|---|---|\n'
    for idx, ex in enumerate(data.get('train', [])[:2]):
        in_md = grid_to_md(ex['input'])
        out_md = grid_to_md(ex['output'])
        md += f'| Train {idx} | {in_md} | {out_md} |\n'
    md += '\n'

with open('/Users/kanxiao/.gemini/antigravity/brain/0f7f4470-3932-41aa-89c0-19c093e6f010/task_visualizations.md', 'w') as f:
    f.write(md)
