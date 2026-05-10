import json
from typing import Awaitable, Callable, List, Optional, Tuple

from mcp import ClientSession
from openai import AsyncOpenAI

from ..config import DEEPSEEK_DEFAULT_BASE_URL
from ..mcp_stdio import tool_catalog_for_planner

SYSTEM_PROMPT = """你是运行在受限工作区内的编排 Agent。根目录为 ./workspace（相对 MCP 工具路径）。
范式：先做最小观察（例如 ls），再调用工具完成任务；路径一律使用相对 workspace 的路径。
未完成翻译/重打包时，明确说明下一步可用的工具与文件位置。"""

MAX_REACT_STEPS = 14
TOOL_RESULT_CAP = 15_000
PREVIEW_CHARS = 800

Emit = Callable[[dict], Awaitable[None]]


def _assistant_message_to_dict(msg) -> dict:
    d = {"role": "assistant", "content": getattr(msg, "content", None) or ""}
    calls = getattr(msg, "tool_calls", None)
    if calls:
        d["tool_calls"] = []
        for tc in calls:
            args = getattr(tc.function, "arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            d["tool_calls"].append(
                {
                    "id": tc.id,
                    "type": getattr(tc, "type", None) or "function",
                    "function": {"name": tc.function.name, "arguments": args},
                }
            )
    return d


def _assistant_message_content(msg) -> Optional[str]:
    c = getattr(msg, "content", None)
    if isinstance(c, str) and c.strip():
        return c
    return None


def _normalize_tool_arguments(raw_args) -> dict:
    if raw_args is None:
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"_raw": raw_args}
    return {"_raw": str(raw_args)}


async def _call_mcp_tool(
    session: ClientSession,
    *,
    emit: Emit,
    name: str,
    arguments: dict,
) -> Tuple[str, bool]:
    await emit({"type": "tool_call", "name": name, "arguments": arguments})
    try:
        res = await session.call_tool(name, arguments)
        res_text = "".join(
            c.text for c in res.content if hasattr(c, "text") and c.text is not None
        )
        if len(res_text) > TOOL_RESULT_CAP:
            res_text = (
                res_text[:TOOL_RESULT_CAP] + "\n\n(内容过长已自动截断)"
            )
        preview = res_text[:PREVIEW_CHARS]
        await emit({"type": "tool_result", "name": name, "preview": preview, "ok": True})
        return res_text, True
    except Exception as e:
        err = f"工具执行失败: {e}"
        await emit({"type": "tool_result", "name": name, "preview": err, "ok": False})
        return err, False


async def run_plan_then_execute(
    *,
    client: AsyncOpenAI,
    model: str,
    session: ClientSession,
    user_goal: str,
    openai_tools: list,
    mcp_tool_names: set,
    emit: Emit,
) -> None:
    """LLM 'compiler' style: single planning JSON, then deterministic tool execution."""
    _ = openai_tools
    tools_data = await session.list_tools()
    catalog = tool_catalog_for_planner(tools_data.tools)
    plan_msgs = [
        {
            "role": "system",
            "content": (
                "你是任务分解器。只输出严格 JSON（不要 markdown），格式：\n"
                '{"steps":[{"tool":"工具名","arguments":{...}},...]}\n'
                "steps 中的 tool 必须来自下列可用工具名；arguments 必须符合该工具参数 schema。\n"
                "若信息不足，输出 {\"steps\":[],\"need_clarification\":\"...\"}。\n"
                f"可用工具：\n{catalog}"
            ),
        },
        {"role": "user", "content": f"目标：{user_goal}"},
    ]

    await emit({"type": "phase", "phase": "plan"})

    plan_resp = await client.chat.completions.create(
        model=model,
        messages=plan_msgs,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    plan_raw = plan_resp.choices[0].message.content or "{}"
    try:
        plan = json.loads(plan_raw)
    except json.JSONDecodeError:
        await emit({"type": "plan_error", "detail": plan_raw[:2000]})
        return

    await emit({"type": "plan", "plan": plan})

    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        clarify = plan.get("need_clarification")
        if clarify:
            await emit({"type": "assistant", "content": clarify})
        return

    await emit({"type": "phase", "phase": "execute"})

    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("tool")
        arguments = step.get("arguments") or {}
        if not isinstance(name, str) or name not in mcp_tool_names:
            await emit({"type": "execute_skip", "reason": "invalid_tool", "step": step})
            continue
        if not isinstance(arguments, dict):
            arguments = _normalize_tool_arguments(arguments)
        await _call_mcp_tool(session, emit=emit, name=name, arguments=arguments)


async def run_react_turn(
    *,
    client: AsyncOpenAI,
    model: str,
    session: ClientSession,
    history: List[dict],
    openai_tools: list,
    emit: Emit,
) -> List[dict]:
    """Classic ReAct via DeepSeek tool calls; updates and returns chat history."""

    trimmed = trim_history(history)

    for _ in range(MAX_REACT_STEPS):
        response = await client.chat.completions.create(
            model=model,
            messages=trimmed,
            tools=openai_tools if openai_tools else None,
        )
        msg = response.choices[0].message
        trimmed.append(_assistant_message_to_dict(msg))
        assistant_text = _assistant_message_content(msg)
        if assistant_text:
            await emit({"type": "assistant", "content": assistant_text})

        calls = getattr(msg, "tool_calls", None)
        if not calls:
            break

        for tc in calls:
            args = _normalize_tool_arguments(tc.function.arguments)
            res_text, _ok = await _call_mcp_tool(
                session, emit=emit, name=tc.function.name, arguments=args
            )
            trimmed.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": res_text,
                }
            )

    return trim_tail(trimmed)


def trim_history(history: List[dict]) -> List[dict]:
    if not history:
        return [{"role": "system", "content": SYSTEM_PROMPT}]
    if history[0].get("role") == "system":
        return history
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def trim_tail(history: List[dict], max_non_system: int = 24) -> List[dict]:
    if not history:
        return history
    head = history[0] if history[0].get("role") == "system" else None
    body = history[1:] if head else history
    if len(body) <= max_non_system:
        return history
    tail = body[-max_non_system:]
    return [head, *tail] if head else tail


def make_deepseek_client(api_key: str, base_url: Optional[str] = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or DEEPSEEK_DEFAULT_BASE_URL,
    )
