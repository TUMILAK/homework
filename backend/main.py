import json
import os
import uuid
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from .agent.engine import SYSTEM_PROMPT, make_deepseek_client
from .agent.engine import run_plan_then_execute as plan_execute
from .agent.engine import run_react_turn as react_turn
from .agent.engine import trim_tail
from .config import (
    BACKEND_HOST,
    BACKEND_PORT,
    DEEPSEEK_DEFAULT_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    PROJECT_ROOT,
    WORKSPACE_ROOT,
)
from .mcp_stdio import mcp_session, openai_tools_from_session as build_tools
from .workspace_fs import ensure_workspace, resolve_safe

FRONTEND_DIR = PROJECT_ROOT / "frontend" / "web"


class NormalizeApiTrailingSlashMiddleware:
    """Strip trailing `/` on `/api/...` so routes match before root StaticFiles (POST → 405)."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path = scope.get("path") or ""
            if path.startswith("/api") and len(path) > 1 and path.endswith("/"):
                scope["path"] = path.rstrip("/")
        await self.app(scope, receive, send)


app = FastAPI(title="MCP Agent Backend", version="0.2.0")

app.add_middleware(NormalizeApiTrailingSlashMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/catalog")
async def catalog_list(depth: Optional[int] = None):
    ensure_workspace()
    root = Path(WORKSPACE_ROOT)
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if depth is not None and depth >= 0 and rel.count("/") > depth:
                continue
            files.append({"path": rel, "size": path.stat().st_size})
    files.sort(key=lambda x: x["path"])
    return {"workspace": str(WORKSPACE_ROOT), "files": files}


@app.get("/api/catalog/download")
async def catalog_download(path: str):
    target = resolve_safe(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Not a file")
    return FileResponse(
        path=str(target),
        filename=target.name,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"},
    )


_WALLPAPER_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg"}
)
_WALLPAPER_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_WALLPAPER_MAX = 12 * 1024 * 1024
_WALLPAPER_LIST_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})


@app.get("/api/wallpaper/list")
async def wallpaper_list():
    """List image files in workspace/wallpaper/ (same tree as upload)."""
    ensure_workspace()
    root = WORKSPACE_ROOT.resolve()
    wp_dir = (root / "wallpaper").resolve()
    if wp_dir != root and root not in wp_dir.parents:
        raise HTTPException(status_code=500, detail="Invalid wallpaper directory")
    wp_dir.mkdir(parents=True, exist_ok=True)
    files: List[dict] = []
    for p in sorted(wp_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in _WALLPAPER_LIST_SUFFIXES:
            continue
        files.append(
            {
                "path": f"wallpaper/{p.name}",
                "name": p.name,
                "size": p.stat().st_size,
            }
        )
    return {
        "workspace_root": str(root),
        "wallpaper_dir": str(wp_dir),
        "files": files,
    }


def _wallpaper_suffix_from_name(filename: str) -> Optional[str]:
    suf = Path(filename or "").suffix.lower()
    if suf == ".jpeg":
        suf = ".jpg"
    if suf in {".jpg", ".png", ".webp", ".gif"}:
        return suf
    return None


@app.post("/api/wallpaper/upload")
async def wallpaper_upload(file: UploadFile = File(...)):
    """Copy image bytes into WORKSPACE_ROOT/wallpaper/ (same as /api/wallpaper/list)."""
    ensure_workspace()
    root = WORKSPACE_ROOT.resolve()
    wp_dir = (root / "wallpaper").resolve()
    if wp_dir != root and root not in wp_dir.parents:
        raise HTTPException(status_code=500, detail="Invalid wallpaper directory")
    wp_dir.mkdir(parents=True, exist_ok=True)

    raw_name = Path(file.filename or "").name
    suf = _wallpaper_suffix_from_name(raw_name)

    ct = (file.content_type or "").split(";")[0].strip().lower()
    if ct not in _WALLPAPER_TYPES:
        if ct in {"application/octet-stream", "binary/octet-stream"} and suf:
            ct = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}[suf]
        elif not ct and suf:
            ct = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}[suf]
        else:
            raise HTTPException(status_code=400, detail="Only image/jpeg, png, webp, gif")

    if suf is None:
        suf = _WALLPAPER_EXT.get(ct, ".jpg")

    dest_name = f"{uuid.uuid4().hex}{suf}"
    dest = wp_dir / dest_name
    data = await file.read()
    if len(data) > _WALLPAPER_MAX:
        raise HTTPException(status_code=400, detail="File too large (max 12MB)")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    dest.write_bytes(data)
    rel = f"wallpaper/{dest_name}"
    return {"path": rel, "name": dest_name, "bytes": len(data)}


_EPUB_MAX_BYTES = 48 * 1024 * 1024


@app.post("/api/epub/localize")
async def epub_localize(
    file: UploadFile = File(...),
    deepseek_api_key: Optional[str] = Form(None),
    deepseek_base_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    epub_glossary: Optional[str] = Form(None),
    epub_regex_rules: Optional[str] = Form(None),
    epub_polish_second: Optional[str] = Form(None),
):
    """Unpack EPUB, batch-translate segments (JSON) to zh-CN, repackage to workspace/epub_out/."""
    from .epub_cn import localize_epub_to_workspace

    key = (deepseek_api_key or "").strip() or os.getenv("DPSK_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=401, detail="需要 DeepSeek API Key（表单或环境变量 DPSK_API_KEY）")

    fname = (file.filename or "book.epub").strip()
    if not fname.lower().endswith(".epub"):
        raise HTTPException(status_code=400, detail="请上传 .epub 文件")

    data = await file.read()
    if len(data) > _EPUB_MAX_BYTES:
        raise HTTPException(status_code=400, detail="EPUB 超过 48MB")
    if len(data) < 32:
        raise HTTPException(status_code=400, detail="文件无效或过短")

    base = (deepseek_base_url or "").strip() or DEEPSEEK_DEFAULT_BASE_URL
    m = (model or "").strip() or DEEPSEEK_DEFAULT_MODEL
    client = AsyncOpenAI(
        api_key=key,
        base_url=base,
        timeout=httpx.Timeout(300.0, connect=30.0),
    )

    ensure_workspace()
    out_dir = WORKSPACE_ROOT / "epub_out"
    stem = Path(fname).stem
    if epub_polish_second is None:
        polish = True
    else:
        s = (epub_polish_second or "").strip().lower()
        polish = s in ("1", "true", "yes", "on") if s else False
    try:
        rel, nbytes = await localize_epub_to_workspace(
            epub_bytes=data,
            stem=stem,
            client=client,
            model=m,
            out_dir=out_dir,
            glossary=(epub_glossary or "").strip(),
            regex_rules=(epub_regex_rules or "").strip(),
            polish_second=polish,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EPUB 处理失败: {e}") from e

    return {"path": rel, "bytes": nbytes}


_IMAGE_COMIC_MAX_BYTES = 25 * 1024 * 1024
_IMAGE_COMIC_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg"}
)


@app.post("/api/image/comic_translate")
async def image_comic_translate(
    file: UploadFile = File(...),
    deepseek_api_key: Optional[str] = Form(None),
    deepseek_base_url: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    image_glossary: Optional[str] = Form(None),
    image_regex_rules: Optional[str] = Form(None),
):
    """OCR → DeepSeek JSON 汉化 → 在识别框内填底嵌字，输出 workspace/image_out/*.png。"""
    from .image_comic import comic_translate_embed_to_workspace

    key = (deepseek_api_key or "").strip() or os.getenv("DPSK_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=401, detail="需要 DeepSeek API Key（表单或环境变量 DPSK_API_KEY）")

    ct = (file.content_type or "").split(";")[0].strip().lower()
    raw_name = Path(file.filename or "page.png").name
    suf = Path(raw_name).suffix.lower()
    if ct not in _IMAGE_COMIC_TYPES:
        if ct in {"application/octet-stream", "binary/octet-stream"} and suf in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
        }:
            ct = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }[suf if suf != ".jpeg" else ".jpg"]
        else:
            raise HTTPException(status_code=400, detail="请上传 jpeg / png / webp / gif 图片")

    data = await file.read()
    if len(data) > _IMAGE_COMIC_MAX_BYTES:
        raise HTTPException(status_code=400, detail="图片超过 25MB")
    if len(data) < 16:
        raise HTTPException(status_code=400, detail="文件无效或过短")

    base = (deepseek_base_url or "").strip() or DEEPSEEK_DEFAULT_BASE_URL
    m = (model or "").strip() or DEEPSEEK_DEFAULT_MODEL
    client = AsyncOpenAI(
        api_key=key,
        base_url=base,
        timeout=httpx.Timeout(300.0, connect=30.0),
    )

    ensure_workspace()
    out_dir = WORKSPACE_ROOT / "image_out"
    stem = Path(raw_name).stem
    try:
        rel, nbytes = await comic_translate_embed_to_workspace(
            image_bytes=data,
            stem=stem,
            client=client,
            model=m,
            out_dir=out_dir,
            glossary=(image_glossary or "").strip(),
            regex_rules=(image_regex_rules or "").strip(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片汉化嵌字失败: {e}") from e

    return {"path": rel, "bytes": nbytes}


@app.options("/api/image/comic_translate")
async def image_comic_translate_options():
    """Avoid OPTIONS falling through to StaticFiles (405) when CORS skips preflight (no Origin)."""
    return Response(status_code=204)


@app.websocket("/ws/agent")
async def ws_agent(ws: WebSocket):
    await ws.accept()

    ds_key: Optional[str] = None
    ds_base: Optional[str] = None
    ds_model = DEEPSEEK_DEFAULT_MODEL
    history: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        async with mcp_session() as session:
            openai_tools, _ = await build_tools(session)
            tools_list = await session.list_tools()
            allowed_names = {t.name for t in tools_list.tools}

            async def emit(event: dict):
                await ws.send_json(event)

            while True:
                try:
                    raw = await ws.receive_text()
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        await ws.send_json(
                            {"type": "error", "detail": "Invalid JSON envelope"}
                        )
                        continue

                    mtype = msg.get("type")
                    if mtype == "configure":
                        ds_key = (
                            msg.get("deepseek_api_key")
                            or msg.get("dpsk_api_key")
                            or os.getenv("DPSK_API_KEY")
                        )
                        ds_base = msg.get("deepseek_base_url") or msg.get("base_url")
                        ds_model = msg.get("model") or DEEPSEEK_DEFAULT_MODEL
                        if ds_key:
                            await ws.send_json({"type": "configured", "model": ds_model})
                        else:
                            await ws.send_json(
                                {
                                    "type": "configured",
                                    "warning": "未提供 DeepSeek Key；请在设置中填写或使用环境变量 DPSK_API_KEY。",
                                    "model": ds_model,
                                }
                            )
                        continue

                    if mtype in ("clear_history",):
                        history = [{"role": "system", "content": SYSTEM_PROMPT}]
                        await ws.send_json({"type": "history_cleared"})
                        continue

                    if not ds_key:
                        await ws.send_json(
                            {"type": "error", "detail": "尚未配置 DeepSeek API Key"}
                        )
                        continue

                    if mtype in ("chat", "message"):
                        goal = msg.get("content") or ""
                        mode = msg.get("mode") or msg.get("schedule") or "react"
                        if not goal.strip():
                            await ws.send_json(
                                {"type": "error", "detail": "Empty content"}
                            )
                            continue

                        history.append({"role": "user", "content": goal})

                        await emit({"type": "turn_start"})

                        client = make_deepseek_client(ds_key, ds_base)

                        if mode == "plan":
                            await plan_execute(
                                client=client,
                                model=ds_model,
                                session=session,
                                user_goal=goal,
                                openai_tools=openai_tools,
                                mcp_tool_names=allowed_names,
                                emit=emit,
                            )
                            history.append(
                                {
                                    "role": "assistant",
                                    "content": f"[plan 已完成]：{goal[:200]}",
                                }
                            )
                        else:
                            history = await react_turn(
                                client=client,
                                model=ds_model,
                                session=session,
                                history=history,
                                openai_tools=openai_tools,
                                emit=emit,
                            )

                        history = trim_tail(history, max_non_system=28)
                        await emit({"type": "turn_done"})
                        continue

                    await ws.send_json(
                        {"type": "error", "detail": "Unknown message type"}
                    )
                except WebSocketDisconnect:
                    raise
                except Exception as loop_err:
                    try:
                        await ws.send_json(
                            {
                                "type": "error",
                                "detail": f"回合异常（将保持连接）：{loop_err}",
                            }
                        )
                    except Exception:
                        pass

    except WebSocketDisconnect:
        return
    except Exception as fatal:
        try:
            await ws.send_json({"type": "fatal", "detail": str(fatal)})
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass


def attach_static_when_present():
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="web")


attach_static_when_present()


def main():
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=os.getenv("UVICORN_RELOAD", "").lower() in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
