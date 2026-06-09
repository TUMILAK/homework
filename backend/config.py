import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
FRONTEND_DIR = ROOT / "frontend" / "web"
# 与 ciallo agent 共用 workspace（壁纸目录 wallpaper/）
WORKSPACE_ROOT = Path(
    os.getenv("SOUTI_WORKSPACE_ROOT", REPO_ROOT / "workspace")
).resolve()
ANSWERS_DIR = Path(os.getenv("SOUTI_ANSWERS_DIR", ROOT / "answers")).resolve()
DATA_DIR = Path(os.getenv("SOUTI_DATA_DIR", ROOT / "data")).resolve()
UPLOAD_DIR = Path(os.getenv("SOUTI_UPLOAD_DIR", ROOT / "uploads")).resolve()

HOST = os.getenv("SOUTI_HOST", "0.0.0.0")
PORT = int(os.getenv("SOUTI_PORT", "8010"))

# 本地 PaddleOCR
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "ch")
PADDLE_OCR_USE_GPU = os.getenv("PADDLE_OCR_USE_GPU", "false").lower() in (
    "1",
    "true",
    "yes",
)

DEFAULT_PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "anthropic": {
        "label": "Anthropic (OpenAI 兼容网关)",
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
    },
    "custom": {
        "label": "自定义",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "llama3",
    },
}

SOLVE_SYSTEM = """你是专业的搜题助手。用户会提供题目文本或 OCR 识别结果。
请：1) 理解题意；2) 分步骤解答；3) 给出最终答案；4) 必要时说明思路与易错点。
回答使用清晰的中文 Markdown，公式可用 LaTeX。"""

CHAT_SYSTEM = """你是友好的 AI 助手，可以进行日常对话、学习辅导与知识问答。
回答准确、简洁，必要时使用 Markdown 排版。"""
