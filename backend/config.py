import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", PROJECT_ROOT / "workspace")).resolve()
MCP_SERVER_SCRIPT = Path(
    os.getenv("MCP_SERVER_SCRIPT", PROJECT_ROOT / "mcp_server.py")
).resolve()

DEEPSEEK_DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL") or os.getenv("DPSK_MODEL") or "deepseek-v4-pro"

BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
