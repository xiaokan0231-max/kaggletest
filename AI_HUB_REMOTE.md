# AI Hub 中心 API + 多客户端配置

本项目现在支持一台电脑作为协作中心，其他电脑只作为客户端访问同一个 API。

## 1. 中心机配置

编辑仓库根目录的 `ai_hub_config.json`：

```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "public_base_url": "http://192.168.1.23:8000"
  },
  "database": {
    "url": "mysql+pymysql://root:@localhost:3306/ai_collab_db?charset=utf8mb4"
  },
  "workspace": {
    "root": ".",
    "default_project": "rogii"
  }
}
```

`public_base_url` 填中心机在局域网里的 IP。`host=0.0.0.0` 表示服务监听所有网卡，其他机器才能访问。

启动中心服务：

```bash
python ai_collab_hub/run_server.py
```

中心机浏览器访问：

```text
http://192.168.1.23:8000
```

## 2. 客户端机器配置

Windows / 另一台 Mac 拉取同一个仓库后，也编辑 `ai_hub_config.json`：

```json
{
  "api": {
    "host": "127.0.0.1",
    "port": 8000,
    "public_base_url": "http://192.168.1.23:8000"
  },
  "database": {
    "url": "mysql+pymysql://root:@localhost:3306/ai_collab_db?charset=utf8mb4"
  },
  "workspace": {
    "root": ".",
    "default_project": "rogii"
  }
}
```

客户端不需要启动 MySQL，也不需要启动 FastAPI。CLI 会把请求发到 `public_base_url`。

检查连接：

```bash
python ai_collab_hub/ai_client.py config --check
```

正常后即可按原协议工作：

```bash
export AI_HUB_PROJECT=neurogolf
python ai_collab_hub/ai_client.py read --name "Codex"
```

Windows PowerShell 等价写法：

```powershell
$env:AI_HUB_PROJECT = "neurogolf"
python ai_collab_hub/ai_client.py read --name "Codex"
```

## 3. 本机覆盖

如果不想改动 git 里的配置，可以新建 `ai_hub_config.local.json`。它会覆盖 `ai_hub_config.json`，且已加入 `.gitignore`。

环境变量仍然优先级最高：

```bash
export AI_HUB_PUBLIC_BASE_URL=http://192.168.1.23:8000
export AI_HUB_DB_URL='mysql+pymysql://root:@localhost:3306/ai_collab_db?charset=utf8mb4'
```

旧变量 `AI_HUB_URL` 仍可用，但它需要包含 `/api`，例如：

```bash
export AI_HUB_URL=http://192.168.1.23:8000/api
```
