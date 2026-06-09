# Release v1.0.0 — 搜题 Agent 首个可用版本

> 发布日期：2026-06-09  
> 代码仓库：https://github.com/TUMILAK/homework  
> 分支：`main`

---

## 概述

**搜题 Agent (souti_agent)** 是一个独立的搜题与学习辅助 Web 应用，基于 FastAPI 后端 + 静态前端构建，集成本地 PaddleOCR 图片识别，支持多模型 AI 解答、对话聊天、本地文件读取和回答存档。

---

## 新增功能

### 🤖 多模型 AI 支持
- 内置 DeepSeek（`deepseek-v4-pro`）、OpenAI（`gpt-4o`）、Anthropic 兼容网关（`claude-sonnet-4`）三个提供商
- 支持「自定义」提供商，可接入任意 OpenAI 兼容 API
- 前端设置页切换模型，偏好自动保存到 localStorage
- 支持 thinking / reasoning 模式

### 📷 图片搜题
- **单张图片 OCR + 求解**：上传图片 → 本地 OCR 识别 → 编辑确认文字 → AI 分步解答
- **批量图片求解**：一次上传多张图片，逐一 OCR 识别并求解
- **纯 OCR 模式**：仅提取图片文字，不调用 AI

### 🔍 本地 OCR（PaddleOCR）
- 基于 PaddleOCR 的纯本地文字识别，无需联网
- 兼容 PaddleOCR 2.x / 3.x 两套 API
- 支持中/英/日等多语言（通过 `PADDLE_OCR_LANG` 环境变量配置）
- 自动处理 Windows OneDNN 兼容性问题

### 💬 聊天模式
- 工作台支持「搜题」和「聊天」两种模式切换
- 聊天模式可选是否写入存档
- 内置专业搜题和友好聊天两套系统提示词

### 📁 本地文件管理
- 支持 TXT / MD / CSV / JSON / LOG / PDF 文件读取
- 支持递归读取整个文件夹
- 浏览器上传文件到 `data/` 目录
- 读取本地文件后可直接提交 AI 求解

### 📝 回答存档
- 搜题结果按日期自动写入 `answers/YYYY-MM-DD.md`
- 同一天的回答自动追加，按时间排序
- 存档文件列表查看与下载

### 🖼️ 壁纸管理
- 支持上传 JPG / PNG / WebP / GIF 壁纸（限 12MB）
- 壁纸目录与 ciallo agent 共用 workspace
- 壁纸列表查看

---

## 技术架构

| 层 | 技术栈 |
|---|--------|
| 后端框架 | FastAPI + Uvicorn (ASGI) |
| LLM 调用 | OpenAI Python SDK（兼容接口） |
| OCR | PaddleOCR（本地，CPU/GPU） |
| PDF 解析 | PyPDF |
| 前端 | 原生 HTML/CSS/JS + Pico CSS 深色主题 |
| 配置管理 | python-dotenv（`.env` 文件） |

---

## API 端点

| 分类 | 端点 | 说明 |
|------|------|------|
| 健康检查 | `GET /api/health` | 服务状态 |
| 提供商 | `GET /api/providers` | 获取默认提供商列表 |
| 验证 | `POST /api/validate-key` | 验证 API Key |
| 对话 | `POST /api/chat` | 发送对话（搜题/聊天模式） |
| 图片 | `POST /api/solve-image` | 单张图片搜题 |
| 图片 | `POST /api/solve-images-batch` | 批量图片搜题 |
| 图片 | `POST /api/ocr` | 纯 OCR 提取 |
| 文件 | `GET /api/data/list` | 列出 data/ 文件 |
| 文件 | `POST /api/data/read` | 读取单个文件 |
| 文件 | `POST /api/data/read-folder` | 递归读取文件夹 |
| 文件 | `POST /api/data/solve-file` | 读取文件并求解 |
| 文件 | `POST /api/upload-temp` | 上传文件 |
| 存档 | `GET /api/answers/list` | 存档列表 |
| 存档 | `GET /api/answers/download` | 下载存档 |
| 工作空间 | `GET /api/catalog` | 文件目录 |
| 工作空间 | `GET /api/catalog/download` | 下载文件 |
| 壁纸 | `GET /api/wallpaper/list` | 壁纸列表 |
| 壁纸 | `POST /api/wallpaper/upload` | 上传壁纸 |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SOUTI_HOST` | `0.0.0.0` | 监听地址 |
| `SOUTI_PORT` | `8010` | 监听端口 |
| `SOUTI_ANSWERS_DIR` | `souti_agent/answers` | 回答存档目录 |
| `SOUTI_DATA_DIR` | `souti_agent/data` | 题目文件目录 |
| `SOUTI_UPLOAD_DIR` | `souti_agent/uploads` | 临时上传目录 |
| `SOUTI_WORKSPACE_ROOT` | `repo_root/workspace` | 工作空间 |
| `PADDLE_OCR_LANG` | `ch` | OCR 语言 |
| `PADDLE_OCR_USE_GPU` | `false` | GPU 加速 |

---

## 安装与启动

```bash
cd souti_agent
pip install -r requirements.txt
python run_server.py
```

浏览器打开：**http://127.0.0.1:8010**

---

## 已知限制

- PaddleOCR 首次运行需要联网下载模型
- Windows CPU 下 `paddlepaddle` 可能需要从官方 whl 源单独安装
- API Key 由浏览器经服务端转发，不在服务端持久化存储
- 不支持 WebSocket / SSE 流式响应
- 前端暂未做移动端适配

---

## 变更统计

| 指标 | 数值 |
|------|------|
| 新增文件 | 47 |
| 新增代码行 | +7,685 |
| 删除代码行 | -608 |
| Python 模块 | 6 (main, config, llm, ocr_service, file_reader, answer_store) |
| 前端页面 | 4 (index, settings, app.js, app.css) |

---

## 贡献者

- TUMILAK

---

## 许可证

本项目和主仓库保持一致。
