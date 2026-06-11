import sys

import uvicorn

try:
    from .config import PROJECT_ROOT, load_config
except ImportError:
    from config import PROJECT_ROOT, load_config


def main():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    cfg = load_config()
    uvicorn.run(
        "ai_collab_hub.main:app",
        host=cfg["api"]["host"],
        port=int(cfg["api"]["port"]),
        reload=False,
    )


if __name__ == "__main__":
    main()
