"""漫画/图片：OCR → DeepSeek 汉化 → 简单嵌字（流程对齐团子翻译器等「识图-译-嵌字」思路，不捆绑第三方 GUI 源码）。"""

from __future__ import annotations

import io
import math
import re
import textwrap
import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from openai import AsyncOpenAI
from PIL import Image, ImageDraw, ImageFont

from .epub_cn import (
    _apply_regex_rules,
    _call_json_segment_batch,
    _parse_epub_regex_rules,
)

_IMAGE_OCR_SYSTEM = """你是漫画、游戏截图或图片台词译者。用户给你一个 JSON：键 "0","1"... 值为 OCR 抽取的原文（可能为日语、英语、繁体中文等）。
只回复一个 JSON 对象（不要 markdown 代码围栏），键集合与输入完全一致，不得增删键；值为适合嵌回图片的**简体中文**短句，语气自然；可略作压缩以适合窄对话框。
若气泡明显为竖排（窄而高），译文尽量短句、少用冗长从句，便于自上而下嵌字。若某条几乎无自然语言可原样返回。"""

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\mingliu.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def _load_font(size: int):
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _quad_to_xyxy(box: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    xs = [float(p[0]) for p in box]
    ys = [float(p[1]) for p in box]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _median_rgb(img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int]:
    crop = img.crop((x0, y0, x1, y1))
    arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    if arr.size == 0:
        return 255, 255, 255
    flat = arr.reshape(-1, 3)
    med = np.median(flat, axis=0)
    return int(med[0]), int(med[1]), int(med[2])


def _text_color_on(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (20, 20, 20) if lum > 140 else (248, 248, 248)


_rapid_ocr_engine = None


def _get_rapid_ocr():
    global _rapid_ocr_engine
    if _rapid_ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError(
                "未安装 OCR 依赖，请执行: pip install rapidocr-onnxruntime onnxruntime opencv-python-headless numpy pillow"
            ) from e
        _rapid_ocr_engine = RapidOCR()
    return _rapid_ocr_engine


def _run_ocr(np_rgb: np.ndarray) -> List[Tuple[List, str, float]]:
    engine = _get_rapid_ocr()
    result, _t = engine(np_rgb)
    if not result:
        return []
    out: List[Tuple[List, str, float]] = []
    for row in result:
        if not row or len(row) < 2:
            continue
        box, rec = row[0], row[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            text, conf = rec[0], float(rec[1])
        else:
            text, conf = str(rec), 1.0
        text = (text or "").strip()
        if not text or conf < 0.25:
            continue
        out.append((box, text, conf))
    return out


async def _translate_lines_json(
    client: AsyncOpenAI,
    model: str,
    lines: List[str],
    glossary: str,
) -> List[str]:
    if not lines:
        return []
    batch_size = 36
    merged: List[str] = []
    for i in range(0, len(lines), batch_size):
        chunk = lines[i : i + batch_size]
        part = await _call_json_segment_batch(
            client, model, chunk, _IMAGE_OCR_SYSTEM, glossary
        )
        if len(part) != len(chunk):
            part = chunk
        merged.extend(part)
    return merged


def _is_vertical_box(w: int, h: int) -> bool:
    """竖排漫画气泡：明显窄高。"""
    if w < 10:
        return h >= w * 1.12
    return h >= w * 1.22


def _vertical_plain(s: str) -> str:
    """竖排时去掉横排空格与换行（竖排单列/多列由算法重排）。"""
    s = (s or "").strip().replace("\r", "")
    s = re.sub(r"[ \t\u3000]+", "", s)
    return s.replace("\n", "")


def _draw_block_vertical(
    draw: ImageDraw.ImageDraw,
    _img: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    text: str,
    fill_rgb: Tuple[int, int, int],
) -> None:
    t = _vertical_plain(text)
    if not t:
        return
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return
    color = _text_color_on(fill_rgb)
    margin_x = max(2, int(w * 0.06))
    margin_y = max(2, int(h * 0.05))
    inner_w = max(6, w - 2 * margin_x)
    inner_h = max(6, h - 2 * margin_y)
    n = len(t)

    best_layout: Optional[Tuple[int, List[str], int, int]] = None
    max_start = min(44, max(10, int(h * 0.24)))
    for font_size in range(max_start, 5, -1):
        font = _load_font(font_size)
        line_gap = max(0, font_size // 14)
        col_gap = max(1, font_size // 5)
        char_heights: List[int] = []
        char_widths: List[int] = []
        for ch in t:
            bb = draw.textbbox((0, 0), ch, font=font)
            char_widths.append(bb[2] - bb[0])
            char_heights.append(bb[3] - bb[1])
        step = max(char_heights) + line_gap if char_heights else font_size + line_gap
        if step <= 0:
            step = font_size
        max_cpc = max(1, int(inner_h // step))
        ncols = max(1, int(math.ceil(n / max_cpc)))
        col_w = (max(char_widths) if char_widths else font_size) + col_gap
        total_w = ncols * col_w - col_gap
        total_h = max_cpc * step - line_gap
        if total_w <= inner_w and total_h <= inner_h:
            cols: List[str] = []
            i = 0
            while i < n:
                cols.append(t[i : i + max_cpc])
                i += max_cpc
            best_layout = (font_size, cols, col_w, line_gap)
            break

    if best_layout is None:
        font_size, cols, col_w, line_gap = 6, [t], max(6, inner_w // max(1, n)), 0
        font = _load_font(font_size)
    else:
        font_size, cols, col_w, line_gap = best_layout
        font = _load_font(font_size)

    right_inner = x1 - margin_x
    top_inner = y0 + margin_y
    for col in cols:
        col_center_x = right_inner - col_w // 2
        y_cur = float(top_inner)
        for ch in col:
            bb = draw.textbbox((0, 0), ch, font=font)
            cw, chh = bb[2] - bb[0], bb[3] - bb[1]
            tx = col_center_x - cw // 2
            draw.text((tx, y_cur), ch, font=font, fill=color)
            y_cur += chh + line_gap
        right_inner -= col_w


def _draw_block(
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    text: str,
) -> None:
    fill_rgb = _median_rgb(img, x0, y0, x1, y1)
    draw.rectangle([x0, y0, x1, y1], fill=fill_rgb)
    t = (text or "").strip()
    if not t:
        return
    w, h = x1 - x0, y1 - y0
    if w < 8 or h < 8:
        return
    if _is_vertical_box(w, h):
        _draw_block_vertical(draw, img, x0, y0, x1, y1, text, fill_rgb)
        return
    color = _text_color_on(fill_rgb)
    inner_w = max(8, int(w * 0.88))
    inner_h = max(8, int(h * 0.88))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    font_size = max(9, min(46, int(h * 0.42)))
    wrapped = t
    font = _load_font(font_size)
    for _ in range(22):
        chars = max(4, inner_w // max(font_size // 2, 7))
        wrapped = textwrap.fill(t, width=chars, break_long_words=True)
        font = _load_font(font_size)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=3)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= inner_w and th <= inner_h:
            break
        font_size -= 1
        if font_size < 8:
            font_size = 8
            font = _load_font(font_size)
            wrapped = textwrap.fill(t, width=max(4, inner_w // 6), break_long_words=True)
            break

    draw.multiline_text(
        (cx, cy),
        wrapped,
        font=font,
        fill=color,
        spacing=3,
        align="center",
        anchor="mm",
    )


async def comic_translate_embed_to_workspace(
    *,
    image_bytes: bytes,
    stem: str,
    client: AsyncOpenAI,
    model: str,
    out_dir: Path,
    glossary: str = "",
    regex_rules: str = "",
) -> Tuple[str, int]:
    """
    OCR → 正则预处理 → DeepSeek JSON 翻译 → 在原图 bbox 上填底+嵌字 → 保存 PNG 到 workspace/image_out/。
    思路参考团子翻译器（OCR + 翻译 + 嵌字），翻译引擎为本项目 DeepSeek。
    """
    rules = _parse_epub_regex_rules(regex_rules)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fff\-_.]+", "_", stem)[:60] or "img"
    name = f"{safe}_{uuid.uuid4().hex[:8]}.png"
    out_path = out_dir / name

    im = Image.open(io.BytesIO(image_bytes))
    if im.mode in ("RGBA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "P":
            im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        im = bg
    else:
        im = im.convert("RGB")

    np_rgb = np.asarray(im, dtype=np.uint8)
    ocr_rows = _run_ocr(np_rgb)
    if not ocr_rows:
        raise ValueError("OCR 未识别到文字（请换清晰图或检查是否已安装 rapidocr-onnxruntime）")

    regions: List[Tuple[Tuple[int, int, int, int], str]] = []
    texts: List[str] = []
    W, H = im.size
    for box, text, _conf in ocr_rows:
        x0, y0, x1, y1 = _quad_to_xyxy(box)
        pad = max(2, int(min(x1 - x0, y1 - y0) * 0.04))
        x0 = _clamp(x0 - pad, 0, W - 1)
        y0 = _clamp(y0 - pad, 0, H - 1)
        x1 = _clamp(x1 + pad, x0 + 2, W)
        y1 = _clamp(y1 + pad, y0 + 2, H)
        t = _apply_regex_rules(text, rules)
        if not t.strip():
            continue
        regions.append(((x0, y0, x1, y1), t))
        texts.append(t)

    if not texts:
        raise ValueError("预处理后无有效文本")

    order = sorted(range(len(regions)), key=lambda i: (regions[i][0][1] + regions[i][0][3]) / 2)
    regions = [regions[i] for i in order]
    texts = [texts[i] for i in order]

    zh_list = await _translate_lines_json(client, model, texts, glossary)
    if len(zh_list) != len(regions):
        zh_list = texts

    work = im.copy()
    draw = ImageDraw.Draw(work)
    for (bbox, _src), zh in zip(regions, zh_list):
        x0, y0, x1, y1 = bbox
        _draw_block(draw, im, x0, y0, x1, y1, (zh or "").strip() or _src)

    work.save(out_path, format="PNG", optimize=True)
    rel = f"image_out/{name}"
    return rel, out_path.stat().st_size
