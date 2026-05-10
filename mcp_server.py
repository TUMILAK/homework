"""FastMCP stdio server: workspace + generic URL capture (subprocess of backend)."""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp import FastMCP

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

workspace = Path("./workspace").resolve()
catalog_dir = workspace / "catalog"
catalog_dir.mkdir(parents=True, exist_ok=True)
workspace.mkdir(exist_ok=True)

mcp = FastMCP("workspace-agent")


def safe_path(rel: str) -> Path:
    if rel is None:
        raise ValueError("EMPTY")
    rp = Path(rel.strip().replace("\\", "/").lstrip("/"))
    resolved = (workspace / rp).resolve()
    ws = workspace.resolve()
    if resolved == ws or ws in resolved.parents:
        return resolved
    raise ValueError("NO")


@mcp.tool()
def ls() -> list:
    files = []
    for root, _, filenames in os.walk(workspace):
        for name in filenames:
            full = Path(root) / name
            rel_path = os.path.relpath(str(full), str(workspace)).replace("\\", "/")
            files.append(rel_path)
    return sorted(files)


@mcp.tool()
def workspace_file_io(action: str, rel_path: str, content: str = None) -> str:
    try:
        abs_path = safe_path(rel_path)
        if action == "read":
            return abs_path.read_text(encoding="utf-8")
        if action == "write":
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content or "", encoding="utf-8")
            return rel_path
        return "unknown action"
    except Exception as e:
        return str(e)


@mcp.tool()
async def workspace_shell(command: str) -> str:
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        decoded_out = out.decode(errors="replace")
        decoded_err = err.decode(errors="replace")
        return (
            f"cwd: {workspace}\n"
            f"exit_code: {proc.returncode}\n"
            f"stdout:\n{decoded_out}\n"
            f"stderr:\n{decoded_err}"
        )
    except Exception as e:
        return str(e)


async def _simple_http_capture(url: str) -> tuple[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    async with httpx.AsyncClient(http2=False, follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        title = urlparse(url).netloc or "page"
        if resp.status_code != 200:
            return title, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        texts = soup.get_text("\n").splitlines()
        body = "\n".join(line.strip() for line in texts if line.strip())
        return title, body


async def _try_crawl4ai(url: str) -> tuple[str, str]:
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore
    except ImportError:
        raise RuntimeError("crawl4ai_not_installed")

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        markdown = getattr(result, "markdown", None) or ""
        extracted = markdown.strip()
        title = urlparse(url).netloc or "page"
        if extracted:
            return title, extracted
        fallback = getattr(result, "extracted_content", None) or ""
        if fallback:
            return title, str(fallback)
        raise RuntimeError("crawl4ai_empty")


@mcp.tool()
async def deep_crawl_url(url: str, rel_out_path: Optional[str] = None) -> str:
    """抓取单个 URL（优先 crawl4ai，否则 HTML→文本），写入 workspace/catalog。"""
    normalized = url.strip()
    slug = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    if rel_out_path:
        target = catalog_dir / rel_out_path.strip().replace("\\", "/")
    else:
        host = urlparse(normalized).netloc.replace(".", "_") or "site"
        target = catalog_dir / f"{host}_{slug}.md"

    try:
        title, body = await _try_crawl4ai(normalized)
        mode = "crawl4ai"
    except Exception:
        title, body = await _simple_http_capture(normalized)
        mode = "fallback"

    rel = os.path.relpath(str(target), str(workspace)).replace("\\", "/")
    _ = safe_path(rel)

    artifact = (
        f"---\n"
        f"title: {title}\n"
        f"url: {normalized}\n"
        f"mode: {mode}\n"
        f"---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(artifact, encoding="utf-8")
    preview = artifact[:1600]
    return (
        f"保存成功 -> {rel}\n"
        f"模式: {mode}\n"
        f"字符数: {len(body)}\n"
        f"预览:\n{preview}"
    )


if __name__ == "__main__":
    mcp.run()
