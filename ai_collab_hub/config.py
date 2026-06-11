import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import socket

def get_lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "ai_hub_config.json"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "ai_hub_config.local.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "public_base_url": "http://127.0.0.1:8000",
    },
    "database": {
        "url": "mysql+pymysql://root:@localhost:3306/ai_collab_db?charset=utf8mb4",
    },
    "workspace": {
        "root": str(PROJECT_ROOT),
        "default_project": "rogii",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_config() -> Dict[str, Any]:
    """Load project-level hub config, with env vars as the final override.

    Committed defaults live in ai_hub_config.json. Machine-specific overrides can
    live in ai_hub_config.local.json, which should stay out of git.
    """
    cfg = _deep_merge(DEFAULT_CONFIG, _read_json(CONFIG_PATH))
    cfg = _deep_merge(cfg, _read_json(LOCAL_CONFIG_PATH))

    if os.environ.get("AI_HUB_PUBLIC_BASE_URL"):
        cfg["api"]["public_base_url"] = os.environ["AI_HUB_PUBLIC_BASE_URL"].rstrip("/")
    if os.environ.get("AI_HUB_HOST"):
        cfg["api"]["host"] = os.environ["AI_HUB_HOST"]
    if os.environ.get("AI_HUB_PORT"):
        cfg["api"]["port"] = int(os.environ["AI_HUB_PORT"])
    if os.environ.get("AI_HUB_DB_URL"):
        cfg["database"]["url"] = os.environ["AI_HUB_DB_URL"]
    if os.environ.get("AI_HUB_DEFAULT_PROJECT"):
        cfg["workspace"]["default_project"] = os.environ["AI_HUB_DEFAULT_PROJECT"]
    if os.environ.get("AI_HUB_WORKSPACE_ROOT"):
        cfg["workspace"]["root"] = os.environ["AI_HUB_WORKSPACE_ROOT"]
    workspace_root = Path(str(cfg["workspace"]["root"])).expanduser()
    if not workspace_root.is_absolute():
        workspace_root = PROJECT_ROOT / workspace_root
    cfg["workspace"]["root"] = str(workspace_root.resolve())

    return cfg


def api_base_url(config: Optional[Dict[str, Any]] = None) -> str:
    cfg = config or load_config()
    base = cfg["api"].get("public_base_url") or "http://127.0.0.1:8000"
    if cfg["api"].get("host") == "0.0.0.0":
        port = cfg["api"].get("port", 8000)
        lan_ip = get_lan_ip()
        return f"http://{lan_ip}:{port}"
    return base.rstrip()


def api_url(config: Optional[Dict[str, Any]] = None) -> str:
    # AI_HUB_URL is kept for backward compatibility and may include /api.
    legacy = os.environ.get("AI_HUB_URL")
    if legacy:
        return legacy.rstrip("/")
    return f"{api_base_url(config)}/api"


def public_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or load_config()
    db_url = cfg["database"].get("url", "")
    db_host = db_url
    if "@" in db_url:
        db_host = db_url.split("@", 1)[1]
    return {
        "config_path": str(CONFIG_PATH),
        "local_config_path": str(LOCAL_CONFIG_PATH),
        "api": {
            "host": cfg["api"].get("host"),
            "port": cfg["api"].get("port"),
            "public_base_url": api_base_url(cfg),
        },
        "database": {
            "url_masked": db_host,
        },
        "workspace": cfg.get("workspace", {}),
    }
