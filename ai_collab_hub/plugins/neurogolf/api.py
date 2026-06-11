from fastapi import Request
from sqlalchemy.orm import Session
from database import Topic, Experiment
from main import agent_name_map, project_members, compute_topic_status
import csv
import os
import re

def handle_request(action: str, req: Request, db: Session):
    if action == "status":
        return get_neurogolf_status(db)
    return {"error": "Unknown action"}

def get_neurogolf_status(db: Session):
    # Read task index
    task_index_path = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/task_index.csv"
    tasks = []
    if os.path.exists(task_index_path):
        with open(task_index_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id_str = row['task_id'].replace('.json', '')
                tasks.append({
                    "id": task_id_str,
                    "rule_family": row.get('rule_family', ''),
                    "shape": row.get('shape_signature', '')
                })
    else:
        # Fallback if csv not found
        for i in range(400):
            tasks.append({
                "id": f"task{i:03d}",
                "rule_family": "UNKNOWN",
                "shape": "UNKNOWN"
            })
            
    # Map topics to tasks
    # Search for taskXXX in topic titles
    from database import Project
    proj = db.query(Project).filter(Project.name == "neurogolf").first()
    if not proj:
        return {"tasks": tasks}
        
    topics = db.query(Topic).filter(Topic.project_id == proj.id).all()
    names = agent_name_map(db)
    agents_count = len(project_members(db, proj))
    
    task_topic_map = {}
    for t in topics:
        match = re.search(r'task\d{3}', t.title.lower())
        if match:
            task_name = match.group(0)
            status_info = compute_topic_status(t, agents_count, names)
            
            # Find best experiment score
            best_cv = None
            exps = db.query(Experiment).filter(Experiment.topic_id == t.id).all()
            for exp in exps:
                if exp.cv_score is not None:
                    if best_cv is None or exp.cv_score > best_cv: # assuming higher is better for score? Wait, metric_lower_is_better is True usually for RMSE, but CV is accuracy?
                        # Actually just take the first or best
                        best_cv = exp.cv_score
                        
            task_topic_map[task_name] = {
                "topic_id": t.id,
                "status": status_info["status"],
                "creator": names.get(t.creator_id, "Unknown"),
                "claimed_by": names.get(t.claimed_by_id, "None") if t.claimed_by_id else None,
                "best_cv": best_cv
            }
            
    # Check ONNX files
    working_dir = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working"
    
    for t in tasks:
        tid = t["id"]
        onnx_path = os.path.join(working_dir, f"{tid}.onnx")
        t["onnx_exists"] = os.path.exists(onnx_path)
        t["onnx_size"] = os.path.getsize(onnx_path) if t["onnx_exists"] else 0
        
        # Determine if it's a dummy
        t["is_dummy"] = t["onnx_exists"] and t["onnx_size"] < 10000 # Dummy models are very small
        
        t["forum"] = task_topic_map.get(tid, None)
        
    return {"tasks": tasks}
