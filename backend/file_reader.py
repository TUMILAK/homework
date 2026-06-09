"""读取本地文本 / Markdown / PDF；支持递归扫描目录（只读）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

_TEXT_EXT = frozenset({".txt", ".md", ".markdown", ".csv", ".json", ".log"})
_PDF_EXT = frozenset({".pdf"})


def _safe_path(base: Path, rel: str) -> Path:
    rel = (rel or "").strip().replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError("路径不能包含 ..")
    target = (base / rel).resolve()
    root = base.resolve()
    if target != root and root not in target.parents:
        raise ValueError("路径必须在数据目录内")
    return target


def read_text_file(path: Path, *, max_chars: int = 120_000) -> str:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    ext = path.suffix.lower()
    if ext in _PDF_EXT:
        return _read_pdf(path, max_chars=max_chars)
    if ext not in _TEXT_EXT and ext:
        raise ValueError(f"不支持的扩展名: {ext}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n…（已截断，原长 {len(text)} 字符）"
    return text


def _read_pdf(path: Path, *, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("读取 PDF 需要安装 pypdf: pip install pypdf") from e
    reader = PdfReader(str(path))
    parts: List[str] = []
    total = 0
    for page in reader.pages:
        chunk = page.extract_text() or ""
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    text = "\n\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n…（已截断）"
    return text or "（PDF 未提取到文本）"


def list_files_recursive(base: Path, rel_dir: str = "") -> List[str]:
    folder = _safe_path(base, rel_dir or ".")
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    out: List[str] = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in _TEXT_EXT or ext in _PDF_EXT:
            out.append(p.relative_to(base.resolve()).as_posix())
    return out


def read_many_files(
    base: Path, rel_paths: List[str], *, max_chars_each: int = 80_000
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []
    for rel in rel_paths:
        rel = rel.strip()
        if not rel:
            continue
        path = _safe_path(base, rel)
        text = read_text_file(path, max_chars=max_chars_each)
        results.append((rel, text))
    return results


def extract_questions_hint(text: str, *, max_len: int = 8000) -> str:
    """简单清理：去掉过多空行。"""
    t = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(t) > max_len:
        t = t[:max_len] + "\n…（已截断）"
    return t
