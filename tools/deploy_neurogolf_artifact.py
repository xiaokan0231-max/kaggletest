from __future__ import annotations

import argparse
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload and deploy a NeuroGolf task ONNX through AI Hub.")
    parser.add_argument("--hub", required=True, help="AI Hub base URL, e.g. http://192.168.137.215:8000")
    parser.add_argument("--task", required=True, help="Task id, e.g. task015 or 15")
    parser.add_argument("--model", required=True, type=Path, help="Candidate taskXXX.onnx")
    parser.add_argument("--score", required=True, type=float, help="Official local score")
    parser.add_argument("--topic", type=int, default=None, help="Source forum topic id")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--project", default="neurogolf")
    parser.add_argument("--allow-regression", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    url = f"{args.hub.rstrip('/')}/api/project_plugin/{args.project}/deploy"
    data = {
        "task_id": args.task,
        "score": str(args.score),
        "agent_name": args.agent,
        "allow_regression": "true" if args.allow_regression else "false",
    }
    if args.topic is not None:
        data["forum_topic_id"] = str(args.topic)
    with args.model.open("rb") as f:
        files = {"file": (args.model.name, f, "application/octet-stream")}
        res = requests.post(url, data=data, files=files, timeout=600)
    if not res.ok:
        print(res.text)
        res.raise_for_status()
    print(res.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
