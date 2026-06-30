import json
import requests
import time

HUB = "http://127.0.0.1:8000/api/project_plugin/neurogolf"
audit_file = "/Users/kanxiao/IdeaProjects/kaggletest/neurogolf/data/working/audits/neurogolf-5689-51-current-rules-open-artifact.json"

data = json.loads(open(audit_file).read())
ready_rows = [r for r in data["rows"] if r["ready"]]
print(f"Deploying {len(ready_rows)} models...")

t0 = time.time()
ok = fail = 0
for i, cand in enumerate(ready_rows, 1):
    task = cand["task"]
    with open(cand["file"], "rb") as fh:
        r = requests.post(f"{HUB}/deploy",
                          files={"file": (f"{task}.onnx", fh, "application/octet-stream")},
                          data={"task_id": task, "score": f"{cand['points']:.3f}",
                                "agent_name": "Gemini", "allow_regression": "true"},
                          timeout=60)
    if r.status_code == 200:
        ok += 1
    else:
        fail += 1
        print(f"FAILED {task}: {r.text[:100]}")
    if i % 50 == 0 or i == len(ready_rows):
        print(f"Deployed {i}/{len(ready_rows)}, ok={ok}, fail={fail}. Elapsed: {time.time()-t0:.1f}s")

print("Hitting /submit to Kaggle...")
r = requests.post(f"{HUB}/submit", data={"message": "single-source-LB-test-5689", "submitted_by": "Gemini"}, timeout=300)
print(f"/submit -> {r.status_code} {r.text}")
