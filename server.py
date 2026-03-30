import os
import re
import asyncio
import cv2
import subprocess
from pathlib import Path
from fastmcp import FastMCP
os.environ['PPOCR_LOG_LEVEL'] = 'ERROR'
from paddleocr import PaddleOCR

workspace = Path("./workspace").resolve()
workspace.mkdir(exist_ok=True)

mcp = FastMCP("1")
ocr = PaddleOCR(use_angle_cls=True, lang="ch", device="gpu",show_log=False)

def safe_path(path) -> Path:
    safepath = (workspace / path).resolve()
    if not str(safepath).startswith(str(workspace)):
        raise ValueError("NO")
    return safepath

@mcp.tool()
def ls() ->list:
    files = []
    for root, _, filenames in os.walk(workspace):
        for f in filenames:
            rel_path = os.path.relpath(os.path.join(root, f), workspace)
            files.append(rel_path)
    return files

@mcp.tool()
async def ocr_image(path:str) -> dict:
    try:
        abs_path = safe_path(path)
        img = cv2.imread(str(abs_path))
        if img is None:
            return {"error": "fail"}
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: ocr.ocr(img,cls=True))
        raw = ""
        if result and result[0]:
            raw = "".join([line[1][0] for line in result[0]])
        return {"text": raw}
    except Exception as e:
        return {"error": str(e)}
    
@mcp.tool()
def workspace_file_io(action: str, rel_path: str, content: str = None) -> str:
    try:
        abs_path = safe_path(rel_path)
        if action == "read":
            return abs_path.read_text(encoding='utf-8')
        elif action == "write":
            abs_path.write_text(content, encoding='utf-8')
            return f"{rel_path}"
    except Exception as e:
        return f"{str(e)}"

@mcp.tool()
async def workspace_shell(command: str) -> str:
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return f"执行目录: {workspace}\n输出: {stdout.decode()}"
    except Exception as e:
        return f"{str(e)}"

if __name__ == "__main__":
    mcp.run()

