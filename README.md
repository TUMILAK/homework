# 搜题 Agent (souti_agent)

一个独立的搜题 / 学习辅助 Web 应用，基于 **FastAPI** 后端 + **静态前端**，集成本地 **PaddleOCR** 图片识别与多模型 AI 解答。

---

## 功能总览

| 功能 | 说明 |
|------|------|
| 🤖 **多模型支持** | DeepSeek / OpenAI / Anthropic（兼容网关）/ 自定义，前端设置页切换，偏好存 localStorage |
| 🔑 **API Key 管理** | 按提供商分别保存 Key，支持「验证 Key」功能（调用 `/api/validate-key`） |
| 📷 **图片搜题** | 单张图片 OCR 识别 → 编辑确认 → AI 求解；支持**批量图片求解** |
| 🔍 **本地 OCR** | 使用 **PaddleOCR** 纯本地识别，无需联网即可提取图片文字 |
| 📁 **本地文件** | 支持 TXT / MD / CSV / JSON / LOG / PDF，可递归读取文件夹，浏览器上传文件到 data/ |
| 💬 **聊天模式** | 工作台可切换「搜题」/「聊天」两种模式，聊天可选是否存档 |
| 📝 **回答存档** | 搜题结果按日期自动写入 `answers/YYYY-MM-DD.md`，同日追加，可下载 |
| 🖼️ **壁纸功能** | 支持上传/管理 workspace 壁纸（JPG/PNG/WebP/GIF），与 ciallo agent 共用 workspace |

---

## 项目结构

```
souti_agent/
├── backend/                 # FastAPI 后端
│   ├── main.py             # 应用入口，定义所有 API 路由
│   ├── config.py           # 配置管理（环境变量、默认提供商、系统提示词）
│   ├── llm.py              # LLM 调用封装（OpenAI 兼容接口，支持 thinking 模式）
│   ├── ocr_service.py      # PaddleOCR 本地图片文字识别
│   ├── file_reader.py      # 本地文件读取（TXT/MD/PDF 等）
│   ├── answer_store.py     # 按日期追加保存回答
│   └── __init__.py
├── frontend/
│   └── web/                # 静态前端页面（Pico CSS + 深色主题）
├── data/                   # 本地题目文件目录
├── answers/                # 回答存档目录（按日生成 YYYY-MM-DD.md）
├── uploads/                # 临时上传目录
├── workspace/              # 工作空间（壁纸等，与 ciallo agent 共用）
├── run_server.py           # 服务启动入口
├── requirements.txt        # Python 依赖
└── README.md               # 本文件
```

---

## 快速开始

### 1. 安装依赖

```bash
cd souti_agent
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python run_server.py
```

浏览器打开：**http://127.0.0.1:8010**

### 3. 配置环境变量（可选）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `SOUTI_HOST` | 监听地址 | `0.0.0.0` |
| `SOUTI_PORT` | 监听端口 | `8010` |
| `SOUTI_ANSWERS_DIR` | 回答存档目录 | `souti_agent/answers` |
| `SOUTI_DATA_DIR` | 本地题目文件目录 | `souti_agent/data` |
| `SOUTI_UPLOAD_DIR` | 临时上传目录 | `souti_agent/uploads` |
| `SOUTI_WORKSPACE_ROOT` | 工作空间目录 | `repo_root/workspace` |
| `PADDLE_OCR_LANG` | OCR 语言（`ch`/`en`/`japan` 等） | `ch` |
| `PADDLE_OCR_USE_GPU` | 是否启用 GPU | `false` |

---

## API 接口一览

### 服务状态
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查，返回服务状态及目录信息 |

### 提供商 & 验证
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/providers` | 获取默认提供商列表 |
| POST | `/api/validate-key` | 验证 API Key 是否有效 |

### 对话 & 搜题
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 发送对话消息（支持 search/solve/chat 三种模式） |
| POST | `/api/solve-image` | 单张图片搜题（OCR → 编辑确认 → AI 解答） |
| POST | `/api/solve-images-batch` | 批量图片搜题 |
| POST | `/api/ocr` | 纯 OCR 识别（不上传 AI，仅返回文字） |

### 本地数据文件
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/data/list` | 列出 data/ 目录下的可读取文件 |
| POST | `/api/data/read` | 读取单个文件内容 |
| POST | `/api/data/read-folder` | 递归读取整个文件夹，返回合并文本 |
| POST | `/api/data/solve-file` | 读取指定文件并提交 AI 求解 |
| POST | `/api/upload-temp` | 上传文件到 data/ 目录 |

### 回答存档
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/answers/list` | 列出所有存档文件 |
| GET | `/api/answers/download` | 下载指定日期的存档文件 |

### 工作空间 & 壁纸
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/catalog` | 列出 workspace 下所有文件 |
| GET | `/api/catalog/download` | 下载 workspace 中文件 |
| GET | `/api/wallpaper/list` | 列出可用壁纸 |
| POST | `/api/wallpaper/upload` | 上传壁纸（限 12MB） |

---

## OCR 说明

本项目使用 **PaddleOCR** 做本地图片文字识别，首次运行会自动下载模型到用户目录（需联网一次）。

### Windows CPU 安装提示

若 `paddlepaddle` 安装失败，可先单独安装官方包：

```bash
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr opencv-python-headless "numpy<2"
```

### 注意事项

- 启动时会自动设置 `FLAGS_use_onednn=0`、`KMP_DUPLICATE_LIB_OK=TRUE` 等环境变量以规避 Windows 兼容性问题
- PaddleOCR 2.x 和 3.x 的构造参数和返回格式不同，`ocr_service.py` 已做兼容处理
- 如 PaddleOCR 无法安装或运行，可将 OCR 调用替换为视觉模型（`llm.py` 中的 `vision_extract_text`）

---

## 系统提示词

项目内置两类系统提示词（在 `backend/config.py` 中定义）：

- **搜题模式** (`SOLVE_SYSTEM`)：专业的搜题助手，要求理解题意、分步骤解答、给出最终答案
- **聊天模式** (`CHAT_SYSTEM`)：友好的 AI 助手，进行日常对话与知识问答

---

## 技术栈

| 层 | 技术 |
|---|------|
| 后端框架 | FastAPI + Uvicorn |
| LLM 调用 | OpenAI Python SDK（兼容接口） |
| OCR | PaddleOCR（本地） |
| PDF 读取 | PyPDF |
| 前端 | 原生 HTML/CSS/JS + Pico CSS 深色主题 |
| 配置管理 | python-dotenv |

---

## 安全说明

- API Key 由浏览器发往本服务再转发至模型 API，**服务端不写 Key 到磁盘**
- 路径访问有安全校验：禁止 `..` 路径穿越，限制在允许的目录范围内
- 壁纸上传限制 12MB，仅允许图片格式（JPEG/PNG/WebP/GIF）

---

## 许可证

本项目和主仓库保持一致。
