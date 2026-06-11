import json
import os
import glob
import math
import subprocess
import csv

from templates.shrink_bbox import fit_and_generate as shrink_bbox_fit
from templates.shrink_extract import fit_and_generate as shrink_extract_fit
from templates.shrink_fold import fit_and_generate as shrink_fold_fit

def run_batch_solver():
    files = glob.glob('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/task???.json')
    
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as f:
        manifest = json.load(f)
        
    # Get all SHRINK tasks
    shrink_tasks = set()
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1].startswith('SHRINK'):
                shrink_tasks.add(row[0].replace('.json', ''))
        
    solved_count = 0
        
    for f in sorted(files):
        task_id = os.path.basename(f).replace('.json', '')
        
        if task_id not in shrink_tasks:
            continue
            
        # Skip if already deployed
        task_info = manifest.get('tasks', {}).get(task_id, {})
        if 'deployed_score' in task_info:
            continue
            
        with open(f, 'r') as fp:
            data = json.load(fp)
            
        code = shrink_bbox_fit(data, task_id)
        template_name = "shrink_bbox"
        
        if code is None:
            code = shrink_extract_fit(data, task_id)
            template_name = "shrink_extract"
            
        if code is None:
            code = shrink_fold_fit(data, task_id)
            template_name = "shrink_fold"
            
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
