"""图片 OCR：本地 PaddleOCR。"""

# 必须在导入 PaddleOCR (paddle) 之前设置，解决 Windows 上 OneDNN fused_conv2d 报错
import os
os.environ.setdefault('FLAGS_use_onednn', '0')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from __future__ import annotations

import asyncio
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PADDLE_OCR_LANG, PADDLE_OCR_USE_GPU

_engine = None
_engine_lock = threading.Lock()


def _create_paddle_ocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "未安装 PaddleOCR。请在 souti_agent 目录执行：\n"
            "  pip install paddlepaddle paddleocr opencv-python-headless \"numpy<2\"\n"
            "Windows CPU 可参考 README 中的官方 whl 源。"
        ) from e

    # 兼容 PaddleOCR 2.x / 3.x 构造参数差异
    attempts = [
        {
            "use_angle_cls": True,
            "lang": PADDLE_OCR_LANG,
            "show_log": False,
            "use_gpu": PADDLE_OCR_USE_GPU,
        },
        {"lang": PADDLE_OCR_LANG, "use_gpu": PADDLE_OCR_USE_GPU},
        {"lang": PADDLE_OCR_LANG},
        {},
    ]
    last_err: Optional[Exception] = None
    for kw in attempts:
        try:
            return PaddleOCR(**kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"PaddleOCR 初始化失败: {last_err}") from last_err


def get_paddle_ocr():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = _create_paddle_ocr()
        return _engine


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图片（支持 JPG/PNG/BMP/WebP）")
    return img


def _parse_paddle_result(result) -> str:
    lines: List[str] = []
    if result is None:
        return ""

    # PaddleOCR 3.x：单条结果为 dict 或 dict 列表
    if isinstance(result, dict):
        result = [result]

    for page in result:
        if page is None:
            continue
        if isinstance(page, dict):
            for key in ("rec_texts", "texts", "rec_text"):
                val = page.get(key)
                if isinstance(val, list):
                    for t in val:
                        s = str(t).strip()
                        if s:
                            lines.append(s)
                    break
                if isinstance(val, str) and val.strip():
                    lines.append(val.strip())
                    break
            continue

        if not isinstance(page, (list, tuple)):
            continue

        for item in page:
            if not item or not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            rec = item[1]
            text = ""
            if isinstance(rec, (list, tuple)) and rec:
                text = str(rec[0]).strip()
            elif isinstance(rec, str):
                text = rec.strip()
            if text:
                lines.append(text)

    return "\n".join(lines)


def _run_paddle_on_image(image_bytes: bytes) -> str:
    ocr = get_paddle_ocr()
    img = _bytes_to_bgr(image_bytes)

    result = None
    if hasattr(ocr, "ocr"):
        try:
            result = ocr.ocr(img, cls=True)
        except TypeError:
            result = ocr.ocr(img)
    if result is None and hasattr(ocr, "predict"):
        result = ocr.predict(img)

    text = _parse_paddle_result(result)
    if not text.strip():
        raise ValueError("PaddleOCR 未识别到文字，请换更清晰的图片")
    return text


async def ocr_image(
    image_bytes: bytes,
    *,
    mime: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Tuple[str, str]:
    """返回 (识别文本, 方法说明)。api_key 等参数保留以兼容旧调用，OCR 仅用本地 Paddle。"""
    _ = mime, api_key, base_url, model
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, _run_paddle_on_image, image_bytes)
    return text, f"paddleocr({PADDLE_OCR_LANG})"
