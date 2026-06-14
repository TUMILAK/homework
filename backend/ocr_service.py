"""图片 OCR：本地 PaddleOCR（Windows + Paddle 3.3 需 patch 关闭 OneDNN/IR 融合）。"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import List, Optional, Tuple


def _bootstrap_paddle_env() -> None:
    for key, val in {
        "FLAGS_use_mkldnn": "0",
        "FLAGS_use_onednn": "0",
        "FLAGS_enable_onednn": "0",
        "FLAGS_use_dnnl": "0",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",
    }.items():
        os.environ[key] = val


_bootstrap_paddle_env()


def _patch_paddleocr_inference() -> None:
    """
    Paddle 3.3 + PaddleOCR 2.8 在 Windows CPU 上即使用 enable_mkldnn=False
    仍会因 IR 融合走 fused_conv2d/OneDNN 崩溃；须 patch Config。
    """
    try:
        import paddle.inference as inference
        import paddleocr.tools.infer.utility as utility
    except ImportError:
        return

    if getattr(utility, "_souti_patched", False):
        return

    Config = inference.Config
    _orig_switch = Config.switch_ir_optim
    _orig_mem = Config.enable_memory_optim
    _orig_mkldnn = Config.enable_mkldnn
    _orig_create = utility.create_predictor

    def switch_ir_optim(self, enable=True):
        return _orig_switch(self, False)

    def enable_memory_optim(self):
        return None

    def enable_mkldnn(self, *args, **kwargs):
        return None

    Config.switch_ir_optim = switch_ir_optim  # type: ignore[method-assign]
    Config.enable_memory_optim = enable_memory_optim  # type: ignore[method-assign]
    Config.enable_mkldnn = enable_mkldnn  # type: ignore[method-assign]

    def create_predictor(args, mode, logger, model_dir=None):
        args.enable_mkldnn = False
        if hasattr(args, "use_mkldnn"):
            args.use_mkldnn = False
        if model_dir is None:
            return _orig_create(args, mode, logger)
        return _orig_create(args, mode, logger, model_dir)

    utility.create_predictor = create_predictor
    utility._souti_patched = True


_patch_paddleocr_inference()

import cv2
import numpy as np

from .config import PADDLE_OCR_LANG, PADDLE_OCR_USE_GPU

_engine = None
_engine_lock = threading.Lock()


def _apply_paddle_runtime_flags() -> None:
    try:
        import paddle

        if hasattr(paddle, "set_flags"):
            paddle.set_flags(
                {
                    "FLAGS_use_mkldnn": False,
                    "FLAGS_use_onednn": False,
                    "FLAGS_enable_onednn": False,
                }
            )
    except Exception:
        pass


def _create_paddle_ocr():
    _patch_paddleocr_inference()
    _apply_paddle_runtime_flags()
    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise RuntimeError(
            "未安装 PaddleOCR。请在 souti_agent 目录执行：\n"
            '  pip install "paddlepaddle>=3.2" paddleocr opencv-python-headless "numpy<2"'
        ) from e

    kw = {
        "lang": PADDLE_OCR_LANG,
        "show_log": False,
        "enable_mkldnn": False,
        "use_gpu": PADDLE_OCR_USE_GPU,
        "use_angle_cls": True,
    }
    try:
        return PaddleOCR(**kw)
    except TypeError:
        kw.pop("show_log", None)
        return PaddleOCR(
            lang=PADDLE_OCR_LANG,
            enable_mkldnn=False,
            use_gpu=PADDLE_OCR_USE_GPU,
        )


def get_paddle_ocr():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = _create_paddle_ocr()
        return _engine


def reset_paddle_ocr() -> None:
    global _engine
    with _engine_lock:
        _engine = None


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


def _run_ocr_infer(ocr, img: np.ndarray):
    if hasattr(ocr, "ocr"):
        try:
            return ocr.ocr(img, cls=True)
        except TypeError:
            return ocr.ocr(img)
    if hasattr(ocr, "predict"):
        try:
            return ocr.predict(img)
        except TypeError:
            return ocr.predict(input=img)
    raise RuntimeError("PaddleOCR 实例缺少 ocr / predict 方法")


def _run_paddle_on_image(image_bytes: bytes) -> str:
    _patch_paddleocr_inference()
    _apply_paddle_runtime_flags()
    ocr = get_paddle_ocr()
    img = _bytes_to_bgr(image_bytes)

    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            result = _run_ocr_infer(ocr, img)
            text = _parse_paddle_result(result)
            if text.strip():
                return text
            raise ValueError("PaddleOCR 未识别到文字，请换更清晰的图片")
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if attempt == 0 and (
                "onednn" in msg or "mkldnn" in msg or "fused_conv2d" in msg
            ):
                reset_paddle_ocr()
                _patch_paddleocr_inference()
                ocr = get_paddle_ocr()
                continue
            raise

    raise RuntimeError(f"OCR 失败: {last_err}") from last_err


async def ocr_image(
    image_bytes: bytes,
    *,
    mime: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Tuple[str, str]:
    _ = mime, api_key, base_url, model
    loop = asyncio.get_running_loop()
    text = await loop.run_in_executor(None, _run_paddle_on_image, image_bytes)
    return text, f"paddleocr({PADDLE_OCR_LANG},ir_off)"
