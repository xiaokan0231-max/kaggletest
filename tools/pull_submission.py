from __future__ import annotations

import argparse
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the current NeuroGolf submission.zip from AI Hub.")
    parser.add_argument("--hub", required=True, help="AI Hub base URL, e.g. http://192.168.137.215:8000")
    parser.add_argument("--out", required=True, type=Path, help="Output path for submission.zip")
    parser.add_argument("--project", default="neurogolf")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base = args.hub.rstrip("/")
    url = f"{base}/api/project_plugin/{args.project}/submission"
    with requests.get(url, stream=True, timeout=60) as res:
        res.raise_for_status()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with tmp.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
