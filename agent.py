"""
Deprecated entrypoint.

本仓库已将「模型编排 FastAPI」与「前端静态页」拆分：

- API + WebSocket +（可选）静态托管：`python run_server.py`
- MCP FastMCP 原子工具：`mcp_server.py`（stdio，由后端子进程拉起）

Gemini/GUI 合体旧实现已移除；Agent 后端仅接入 DeepSeek（可由前端传入 Key）。
"""
