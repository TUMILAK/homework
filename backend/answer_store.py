"""按日期追加保存回答到 answers/YYYY-MM-DD.md。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def _day_path(answers_dir: Path, day: Optional[str] = None) -> Path:
    answers_dir.mkdir(parents=True, exist_ok=True)
    name = day or datetime.now().strftime("%Y-%m-%d")
    return answers_dir / f"{name}.md"


def append_answer(
    *,
    answers_dir: Path,
    question: str,
    answer: str,
    mode: str = "solve",
    source: str = "",
) -> str:
    """追加一条记录，返回写入的文件相对名。"""
    now = datetime.now()
    path = _day_path(answers_dir, now.strftime("%Y-%m-%d"))
    block = [
        f"\n## {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- 模式：`{mode}`",
    ]
    if source.strip():
        block.append(f"- 来源：{source.strip()}")
    block.extend(
        [
            "",
            "### 问题",
            "",
            question.strip() or "（无）",
            "",
            "### 回答",
            "",
            answer.strip() or "（无）",
            "",
            "---",
            "",
        ]
    )
    header = ""
    if not path.is_file():
        header = f"# 搜题记录 {now.strftime('%Y-%m-%d')}\n\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(header + "\n".join(block))
    return path.name
