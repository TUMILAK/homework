"""搜题 Agent 后端：FastAPI + 静态前端。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answer_store import append_answer
from .config import ANSWERS_DIR, DATA_DIR, DEFAULT_PROVIDERS, FRONTEND_DIR, UPLOAD_DIR, WORKSPACE_ROOT
from .file_reader import extract_questions_hint, list_files_recursive, read_many_files, read_text_file
from .llm import chat_completion, validate_api_key
from .ocr_service import ocr_image

for d in (ANSWERS_DIR, DATA_DIR, UPLOAD_DIR, WORKSPACE_ROOT, WORKSPACE_ROOT / "wallpaper"):
    d.mkdir(parents=True, exist_ok=True)

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


def _resolve_workspace(rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="非法路径")
    target = (WORKSPACE_ROOT / rel).resolve()
    root = WORKSPACE_ROOT.resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="路径必须在 workspace 内")
    return target


def _wallpaper_suffix_from_name(filename: str) -> Optional[str]:
    suf = Path(filename or "").suffix.lower()
    if suf == ".jpeg":
        suf = ".jpg"
    if suf in {".jpg", ".png", ".webp", ".gif"}:
        return suf
    return None

app = FastAPI(title="搜题 Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValidateKeyBody(BaseModel):
    api_key: str
    base_url: str
    model: str


class ChatMessage(BaseModel):
    role: str
    content: str
    reasoning_content: Optional[str] = None


class ChatBody(BaseModel):
    api_key: str
    base_url: str
    model: str
    messages: List[ChatMessage] = Field(default_factory=list)
    mode: str = "solve"
    save_answer: bool = True
    source: str = ""
    use_thinking: bool = True


class ReadFileBody(BaseModel):
    path: str


class ReadFolderBody(BaseModel):
    folder: str = ""
    max_files: int = 20


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "service": "souti_agent",
        "answers_dir": str(ANSWERS_DIR),
        "data_dir": str(DATA_DIR),
        "workspace_root": str(WORKSPACE_ROOT),
    }


@app.get("/api/catalog")
async def catalog_list(depth: Optional[int] = None) -> dict:
    root = WORKSPACE_ROOT.resolve()
    files: List[dict] = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if depth is not None and depth >= 0 and rel.count("/") > depth:
                continue
            files.append({"path": rel, "size": path.stat().st_size})
    files.sort(key=lambda x: x["path"])
    return {"workspace": str(root), "files": files}


@app.get("/api/catalog/download")
async def catalog_download(path: str):
    target = _resolve_workspace(path)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        path=str(target),
        filename=target.name,
        headers={"Access-Control-Expose-Headers": "Content-Disposition"},
    )


@app.get("/api/wallpaper/list")
async def wallpaper_list() -> dict:
    root = WORKSPACE_ROOT.resolve()
    wp_dir = (root / "wallpaper").resolve()
    if wp_dir != root and root not in wp_dir.parents:
        raise HTTPException(status_code=500, detail="Invalid wallpaper directory")
    wp_dir.mkdir(parents=True, exist_ok=True)
    files: List[dict] = []
    for p in sorted(wp_dir.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _WALLPAPER_LIST_SUFFIXES:
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


@app.post("/api/wallpaper/upload")
async def wallpaper_upload(file: UploadFile = File(...)) -> dict:
    root = WORKSPACE_ROOT.resolve()
    wp_dir = (root / "wallpaper").resolve()
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


@app.get("/api/providers")
async def providers() -> dict:
    return {"providers": DEFAULT_PROVIDERS}


@app.post("/api/validate-key")
async def api_validate_key(body: ValidateKeyBody) -> dict:
    key = (body.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API Key 不能为空")
    return await validate_api_key(
        api_key=key,
        base_url=body.base_url,
        model=body.model,
    )


@app.post("/api/chat")
async def api_chat(body: ChatBody) -> dict:
    key = (body.api_key or "").strip()
    if not key:
        raise HTTPException(status_code=401, detail="需要 API Key")
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")

    mode = "chat" if body.mode == "chat" else "solve"
    msgs = [m.model_dump() for m in body.messages]
    question = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            question = str(m.get("content") or "")
            break

    try:
        answer = await chat_completion(
            api_key=key,
            base_url=body.base_url,
            model=body.model,
            messages=msgs,
            mode=mode,
            use_thinking=body.use_thinking,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    saved_file = ""
    if body.save_answer and mode == "solve":
        saved_file = append_answer(
            answers_dir=ANSWERS_DIR,
            question=question,
            answer=answer,
            mode=mode,
            source=body.source,
        )

    return {"answer": answer, "mode": mode, "saved_file": saved_file}


@app.post("/api/ocr")
async def api_ocr(
    file: UploadFile = File(...),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    mime = file.content_type or "image/jpeg"
    try:
        text, method = await ocr_image(
            data,
            mime=mime,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"text": text, "method": method, "filename": file.filename or "image"}


@app.post("/api/solve-image")
async def api_solve_image(
    api_key: str = Form(...),
    base_url: str = Form(""),
    model: str = Form(...),
    save_answer: bool = Form(True),
    confirm_text: str = Form(""),
    file: Optional[UploadFile] = File(None),
) -> dict:
    fname = "preview"
    if confirm_text.strip():
        ocr_text = confirm_text.strip()
        method = "user-confirmed"
    else:
        if file is None:
            raise HTTPException(status_code=400, detail="请上传图片或填写 OCR 预览文本")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="空文件")
        mime = file.content_type or "image/jpeg"
        fname = file.filename or "image"
        try:
            ocr_text, method = await ocr_image(
                data,
                mime=mime,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

    question = f"[图片题目 OCR]\n文件名：{fname}\n\n{ocr_text}"
    try:
        answer = await chat_completion(
            api_key=api_key.strip(),
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": question}],
            mode="solve",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    saved_file = ""
    if save_answer:
        saved_file = append_answer(
            answers_dir=ANSWERS_DIR,
            question=question,
            answer=answer,
            mode="solve",
            source=f"image:{fname}",
        )

    return {
        "ocr_text": ocr_text,
        "ocr_method": method,
        "answer": answer,
        "saved_file": saved_file,
    }


@app.post("/api/solve-images-batch")
async def api_solve_images_batch(
    files: List[UploadFile] = File(...),
    api_key: str = Form(...),
    base_url: str = Form(""),
    model: str = Form(...),
    save_answer: bool = Form(True),
) -> dict:
    results: List[Dict[str, Any]] = []
    for f in files:
        item: Dict[str, Any] = {"filename": f.filename or "image"}
        try:
            data = await f.read()
            mime = f.content_type or "image/jpeg"
            ocr_text, method = await ocr_image(
                data,
                mime=mime,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            question = f"[批量图片]\n{f.filename}\n\n{ocr_text}"
            answer = await chat_completion(
                api_key=api_key.strip(),
                base_url=base_url,
                model=model,
                messages=[{"role": "user", "content": question}],
                mode="solve",
            )
            saved = ""
            if save_answer:
                saved = append_answer(
                    answers_dir=ANSWERS_DIR,
                    question=question,
                    answer=answer,
                    mode="solve",
                    source=f"batch:{f.filename}",
                )
            item.update(
                {
                    "ok": True,
                    "ocr_text": ocr_text,
                    "ocr_method": method,
                    "answer": answer,
                    "saved_file": saved,
                }
            )
        except Exception as e:
            item.update({"ok": False, "error": str(e)})
        results.append(item)
    return {"count": len(results), "results": results}


@app.get("/api/data/list")
async def api_data_list(folder: str = "") -> dict:
    try:
        files = list_files_recursive(DATA_DIR, folder)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"folder": folder or ".", "files": files, "data_dir": str(DATA_DIR)}


@app.post("/api/data/read")
async def api_data_read(body: ReadFileBody) -> dict:
    try:
        from .file_reader import _safe_path

        path = _safe_path(DATA_DIR, body.path)
        text = read_text_file(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"path": body.path, "text": text}


@app.post("/api/data/read-folder")
async def api_data_read_folder(body: ReadFolderBody) -> dict:
    try:
        all_files = list_files_recursive(DATA_DIR, body.folder)
        selected = all_files[: max(1, min(body.max_files, 50))]
        chunks = read_many_files(DATA_DIR, selected)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    combined = "\n\n".join(
        f"=== {rel} ===\n{extract_questions_hint(text)}" for rel, text in chunks
    )
    return {
        "folder": body.folder or ".",
        "file_count": len(chunks),
        "files": [c[0] for c in chunks],
        "combined_text": combined,
    }


@app.post("/api/data/solve-file")
async def api_data_solve_file(
    path: str = Form(...),
    api_key: str = Form(...),
    base_url: str = Form(""),
    model: str = Form(...),
    save_answer: bool = Form(True),
) -> dict:
    try:
        from .file_reader import _safe_path

        fp = _safe_path(DATA_DIR, path)
        text = read_text_file(fp)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    question = f"[本地文件] {path}\n\n{extract_questions_hint(text)}"
    try:
        answer = await chat_completion(
            api_key=api_key.strip(),
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": question}],
            mode="solve",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    saved_file = ""
    if save_answer:
        saved_file = append_answer(
            answers_dir=ANSWERS_DIR,
            question=question,
            answer=answer,
            mode="solve",
            source=f"file:{path}",
        )
    return {"path": path, "answer": answer, "saved_file": saved_file}


@app.post("/api/upload-temp")
async def api_upload_temp(file: UploadFile = File(...)) -> dict:
    """浏览器上传文件到 data/ 目录（只写入，不修改已有同名以外的逻辑）。"""
    name = Path(file.filename or "upload.txt").name
    if ".." in name:
        raise HTTPException(status_code=400, detail="非法文件名")
    dest = DATA_DIR / name
    data = await file.read()
    dest.write_bytes(data)
    rel = dest.relative_to(DATA_DIR).as_posix()
    return {"path": rel, "size": len(data)}


@app.get("/api/answers/list")
async def api_answers_list() -> dict:
    files = sorted(
        [p.name for p in ANSWERS_DIR.glob("*.md") if p.is_file()],
        reverse=True,
    )
    return {"files": files, "answers_dir": str(ANSWERS_DIR)}


@app.get("/api/answers/download")
async def api_answers_download(name: str):
    safe = Path(name).name
    if not safe.endswith(".md"):
        raise HTTPException(status_code=400, detail="仅支持 .md")
    path = ANSWERS_DIR / safe
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path, media_type="text/markdown", filename=safe)


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
