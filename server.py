import os
import re
import asyncio
import cv2
import subprocess
from pathlib import Path
from fastmcp import FastMCP
os.environ['PPOCR_LOG_LEVEL'] = 'ERROR'
from paddleocr import PaddleOCR
import httpx
from bs4 import BeautifulSoup

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


@mcp.tool()
async def crawl_syosetu_chapter(novel_code: str, chapter_num: str) -> str:
    url = f"https://ncode.syosetu.com/{novel_code}/{chapter_num}/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": f"https://ncode.syosetu.com/{novel_code}/",
    }
    cookies = {"over18": "yes"}

    try:
        async with httpx.AsyncClient(http2=False, follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers=headers, cookies=cookies)
            
            if response.status_code != 200:
                return f"错误：无法访问页面，状态码 {response.status_code}"

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 兼容性选择器
            title_node = soup.select_one('h1.p-novel__title, .novel_subtitle')
            title = title_node.get_text(strip=True) if title_node else "无标题"

            content_div = soup.select_one('div.p-novel__body, #novel_honbun')
            if not content_div:
                return "错误：未能解析到正文内容，请检查小说代码或章节号。"

            paragraphs = content_div.find_all('p')
            content_text = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]

            result = f"【章节标题】：{title}\n"
            result += "-" * 20 + "\n"
            result += "\n\n".join(content_text)
            
            return result

    except Exception as e:
        import traceback
        return f"{traceback.format_exc()}"


if __name__ == "__main__":
    mcp.run()

