#!/usr/bin/env python3
"""启动搜题 Agent 服务（默认 http://127.0.0.1:8010）。"""

import os
import sys
from pathlib import Path

# Paddle 须在 import 前关闭 OneDNN，避免 Windows CPU 上 fused_conv2d 崩溃
for _k, _v in {
    "FLAGS_use_mkldnn": "0",
    "FLAGS_use_onednn": "0",
    "FLAGS_enable_onednn": "0",
    "FLAGS_use_dnnl": "0",
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
}.items():
    os.environ[_k] = _v

import uvicorn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)
