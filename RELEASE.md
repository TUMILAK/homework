# Release v1.1.0 — OCR 识别修复与 Paddle 3.3 兼容

> 发布日期：2026-06-14  
> 代码仓库：https://github.com/TUMILAK/homework  
> 分支：`main`  
> 上一版本：v1.0.0

---

## 概述

本次更新主要修复了 **PaddleOCR 在 Windows CPU 环境下的图片识别崩溃问题**。针对 Paddle 3.3 + PaddleOCR 2.8 的组合重写了 `ocr_service.py`，通过底层 patch 彻底解决 `fused_conv2d` / OneDNN IR 融合导致的崩溃。

---

## 🔧 修复内容

### OCR 引擎重写（`backend/ocr_service.py`）

| 修复项 | 说明 |
|--------|------|
| **环境变量引导** | `_bootstrap_paddle_env()` 在模块导入前就关闭所有 OneDNN/MKLDNN/DNNL 相关 FLAGS，新增 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` 等变量 |
| **底层 Inference Patch** | `_patch_paddleocr_inference()` monkey-patch `paddle.inference.Config` 和 `paddleocr.tools.infer.utility.create_predictor`，强制关闭 IR 图优化、内存优化和 MKLDNN |
| **Paddle Runtime Flags** | `_apply_paddle_runtime_flags()` 在 PaddleOCR 初始化前通过 `paddle.set_flags()` 二次确保关闭 OneDNN |
| **OCR 引擎重置** | 新增 `reset_paddle_ocr()` 函数支持运行时重新初始化引擎 |
| **版本约束** | PaddleOCR 初始化参数统一为当前兼容写法，移除多重 fallback 尝试 |

### 启动脚本增强（`run_server.py`）

- 在导入所有模块之前就设置好 Paddle 环境变量（`FLAGS_use_mkldnn=0` 等），防止 import 阶段触发 Paddle 初始化崩溃

### 依赖版本锁定（`requirements.txt`）

```
paddlepaddle>=3.2.0,<3.4.0   # 锁定 3.2-3.3
paddleocr>=2.7.0,<3.0.0      # 锁定 2.x
```

---

## 📊 变更统计

| 指标 | 数值 |
|------|------|
| 修改文件 | 3 |
| 新增代码行 | +159 |
| 删除代码行 | -45 |
| 影响模块 | `ocr_service.py`（重写）、`run_server.py`、`requirements.txt` |

---

## 🐛 已修复问题

- **Windows CPU 环境下 `fused_conv2d` 崩溃**：Paddle 3.3 在 Windows CPU 上即使显式设置 `enable_mkldnn=False`，IR 融合仍会走 OneDNN 路径导致崩溃。本次通过底层 monkey-patch `Config` 类彻底阻断
- **import 阶段崩溃**：Paddle 在 import 时就可能初始化 FLAGS，`run_server.py` 现在在 import 前设置环境变量
- **不稳定的初始化参数**：旧版用多重 `try/except` 尝试不同参数，新版统一为确定的参数组合

---

## 📥 升级指南

```bash
cd souti_agent
git pull origin main
pip install -r requirements.txt
python run_server.py
```

无需修改配置文件或数据目录，直接升级即可。

---

## 贡献者

- TUMILAK

---

## 许可证

本项目和主仓库保持一致。
