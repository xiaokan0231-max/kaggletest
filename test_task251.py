from neurogolf_gemini.templates.fill_holes import fit_and_generate
import json
import subprocess
import os
import sys

with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task251.json') as f:
    data = json.load(f)
    
code = fit_and_generate(data, 'task251')
if code:
    script_path = '/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/solve_251_auto.py'
    with open(script_path, 'w') as f:
        f.write(code)
    subprocess.run(['python3', script_path], check=True)
    
    verify_script = f"""import sys
sys.path.append('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils
neurogolf_utils.verify_network('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task251.onnx', 'task251')
"""
    with open('/tmp/verify_251.py', 'w') as vf:
        vf.write(verify_script)
    subprocess.run(['python3', '/tmp/verify_251.py'], check=True)
