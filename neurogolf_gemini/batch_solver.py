import json
import os
import glob
import math
import subprocess

from templates.color_map import fit_and_generate as color_map_fit
from templates.translation import fit_and_generate as translation_fit
from templates.fill_holes import fit_and_generate as fill_holes_fit
from templates.morphology import fit_and_generate as morphology_fit
from templates.ray_casting import fit_and_generate as ray_casting_fit

def run_batch_solver():
    files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task???.json')
    
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    solved_count = 0
        
    for f in sorted(files):
        task_id = os.path.basename(f).replace('.json', '')
        # Skip if already deployed
        task_info = manifest.get('tasks', {}).get(task_id, {})
        if 'deployed_score' in task_info:
            continue
            
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        code = color_map_fit(data, task_id)
        template_name = "color_map"
        if code is None:
            code = translation_fit(data, task_id)
            template_name = "translation"
        if code is None:
            code = fill_holes_fit(data, task_id)
            template_name = "fill_holes"
        if code is None:
            code = morphology_fit(data, task_id)
            template_name = "morphology"
        if code is None:
            code = ray_casting_fit(data, task_id)
            template_name = "ray_casting"
            
        if code is not None:
            print(f"[{task_id}] Fitted with {template_name} template!")
            # Save the code to a temporary file
            script_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/solve_{task_id[-3:]}_auto.py"
            with open(script_path, 'w') as out_f:
                out_f.write(code)
            
            # Execute the script to generate ONNX
            try:
                subprocess.run(['python3', script_path], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"[{task_id}] Error generating ONNX: {e}")
                continue
                
            onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/{task_id}.onnx"
            
            # Verify network
            try:
                # Call verify_network as a subprocess
                verify_script = f"""import sys
import os
sys.path.append('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils
import onnx
try:
    neurogolf_utils.verify_network('{onnx_path}', '{task_id}')
except Exception as e:
    print(e)
    sys.exit(1)
"""
                with open('/tmp/verify_tmp.py', 'w') as vf:
                    vf.write(verify_script)
                
                res = subprocess.run(['python3', '/tmp/verify_tmp.py'], check=True, capture_output=True, text=True)
                
                if "Your network IS READY for submission!" in res.stdout:
                    print(f"[{task_id}] VERIFICATION PASSED!")
                    # Deploy to Hub
                    size = os.path.getsize(onnx_path)
                    score = max(1.0, 25.0 - math.log(max(1.0, size)))
                    deploy_cmd = ['python3', '/Users/kanxiao/IdeaProjects/kaggletest/tools/deploy_neurogolf_artifact.py',
                                  '--hub', 'http://192.168.137.215:8000',
                                  '--task', task_id,
                                  '--model', onnx_path,
                                  '--score', f"{score:.3f}",
                                  '--agent', 'Gemini_Auto',
                                  '--allow-regression']
                    deploy_res = subprocess.run(deploy_cmd, check=True, capture_output=True, text=True)
                    print(f"[{task_id}] DEPLOYED TO HUB!")
                    solved_count += 1
                else:
                    print(f"[{task_id}] Verification failed: output did not contain success message.")
                    print(res.stdout)
            except subprocess.CalledProcessError as e:
                print(f"[{task_id}] Verification failed with exception.")
                print(e.stdout)
                
    print(f"Batch solver finished! Successfully solved and deployed {solved_count} tasks.")

if __name__ == '__main__':
    run_batch_solver()
