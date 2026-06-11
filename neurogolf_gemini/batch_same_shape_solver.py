import json
import os
import glob
import math
import subprocess
import multiprocessing

def process_task(task_id):
    f = f'/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/{task_id}.json'
    
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/solution_manifest.json', 'r') as mf:
        manifest = json.load(mf)
        
    if 'deployed_score' in manifest.get('tasks', {}).get(task_id, {}):
        return None
        
    with open(f, 'r') as fp:
        data = json.load(fp)
        
    # Import templates here to ensure multiprocess safety
    from neurogolf_gemini.templates.color_map import fit_and_generate as color_map_fit
    from neurogolf_gemini.templates.translation import fit_and_generate as translation_fit
    from neurogolf_gemini.templates.fill_holes import fit_and_generate as fill_holes_fit
    from neurogolf_gemini.templates.morphology import fit_and_generate as morphology_fit
    from neurogolf_gemini.templates.ray_casting import fit_and_generate as ray_casting_fit
    from neurogolf_gemini.templates.gravity_drop import fit_and_generate as gravity_drop_fit
    from neurogolf_gemini.templates.symmetry_reflection import fit_and_generate as symmetry_reflection_fit
    
    templates = [
        ('color_map', color_map_fit),
        ('translation', translation_fit),
        ('fill_holes', fill_holes_fit),
        ('morphology', morphology_fit),
        ('ray_casting', ray_casting_fit),
        ('gravity_drop', gravity_drop_fit),
        ('symmetry_reflection', symmetry_reflection_fit)
    ]
    
    for name, func in templates:
        try:
            code = func(data, task_id)
            if code is not None:
                return task_id, name, code
        except Exception as e:
            pass
            
    return None

def verify_and_deploy(task_id, name, code):
    print(f"[{task_id}] Fitted with {name} template!")
    script_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf_gemini/solve_{task_id[-3:]}_auto.py"
    with open(script_path, 'w') as out_f:
        out_f.write(code)
    
    try:
        subprocess.run(['python3', script_path], check=True, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        print(f"[{task_id}] Error generating ONNX: Timeout")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[{task_id}] Error generating ONNX: {e}")
        return False
        
    onnx_path = f"/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/{task_id}.onnx"
    
    verify_script = f"""import sys
sys.path.append('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/raw/neurogolf_utils')
import neurogolf_utils
try:
    neurogolf_utils.verify_network('{onnx_path}', '{task_id}')
except Exception as e:
    print(e)
    sys.exit(1)
"""
    tmp_verify = f'/tmp/verify_{task_id}.py'
    with open(tmp_verify, 'w') as vf:
        vf.write(verify_script)
        
    try:
        res = subprocess.run(['python3', tmp_verify], check=True, capture_output=True, text=True, timeout=60)
        if "Your network IS READY for submission!" in res.stdout:
            print(f"[{task_id}] VERIFICATION PASSED!")
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
            return True
        else:
            print(f"[{task_id}] Verification failed: output did not contain success message.")
    except subprocess.TimeoutExpired:
        print(f"[{task_id}] Verification failed: Timeout")
    except subprocess.CalledProcessError as e:
        print(f"[{task_id}] Verification failed with exception.")
        
    return False

def run_batch_solver():
    import csv
    tasks_to_run = []
    with open('/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[1] == 'SAME_SHAPE':
                tasks_to_run.append(row[0].replace('.json', ''))
                
    solved_count = 0
    pool = multiprocessing.Pool(processes=8)
    results = pool.imap_unordered(process_task, tasks_to_run)
    
    for res in results:
        if res is not None:
            task_id, name, code = res
            if verify_and_deploy(task_id, name, code):
                solved_count += 1
                
    pool.close()
    pool.join()
    
    print(f"Batch solver finished! Successfully solved and deployed {solved_count} SAME_SHAPE tasks.")

if __name__ == '__main__':
    run_batch_solver()
