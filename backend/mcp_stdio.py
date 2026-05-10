import json
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCP_SERVER_SCRIPT, PROJECT_ROOT


def clean_schema(schema):
    if not isinstance(schema, dict):
        return schema
    return {
        k: clean_schema(v)
        for k, v in schema.items()
        if k not in ("additional_properties", "additionalProperties")
    }


@asynccontextmanager
async def mcp_session():
    """One MCP stdio session (FastMCP server subprocess)."""
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT)],
        cwd=str(PROJECT_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def openai_tools_from_session(session: ClientSession):
    tools_data = await session.list_tools()
    openai_tools = []
    for t in tools_data.tools:
        decl = {
            "name": t.name,
            "description": t.description,
            "parameters": clean_schema(t.inputSchema),
        }
        openai_tools.append({"type": "function", "function": decl})
    return openai_tools, tools_data.tools


def tool_catalog_for_planner(tools) -> str:
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description or ''}")
    return "\n".join(lines)
