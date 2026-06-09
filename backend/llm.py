"""OpenAI 兼容 Chat Completions（支持 DeepSeek thinking 参数）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .config import CHAT_SYSTEM, SOLVE_SYSTEM

THINKING_KW = {
    "reasoning_effort": "high",
    "extra_body": {"thinking": {"type": "enabled"}},
}


def make_client(api_key: str, base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=(base_url or "").strip().rstrip("/") or None,
    )


async def validate_api_key(
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> Dict[str, Any]:
    client = make_client(api_key, base_url)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            **THINKING_KW,
        )
        content = (resp.choices[0].message.content or "").strip()
        return {"ok": True, "message": "API Key 有效", "sample": content[:200]}
    except Exception as e:
        return {"ok": False, "message": str(e)}


async def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: List[dict],
    mode: str = "solve",
    use_thinking: bool = True,
) -> str:
    client = make_client(api_key, base_url)
    system = SOLVE_SYSTEM if mode == "solve" else CHAT_SYSTEM
    full: List[dict] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            full.append({"role": role, "content": content})
            rc = m.get("reasoning_content")
            if rc and full[-1]["role"] == "assistant":
                full[-1]["reasoning_content"] = rc

    kw: dict = {"model": model, "messages": full}
    if use_thinking:
        kw.update(THINKING_KW)

    resp = await client.chat.completions.create(**kw)
    msg = resp.choices[0].message
    return (msg.content or "").strip()


async def vision_extract_text(
    *,
    api_key: str,
    base_url: str,
    model: str,
    image_b64: str,
    mime: str = "image/jpeg",
) -> str:
    """用支持视觉的模型从图片提取题目文字。"""
    client = make_client(api_key, base_url)
    data_url = f"data:{mime};base64,{image_b64}"
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请识别图片中的全部题目文字，按阅读顺序输出纯文本，不要解答。",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()
