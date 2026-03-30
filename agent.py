import os
import json
import asyncio
from openai import OpenAI
from fastmcp import Client
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
key = os.getenv("DPSK_API_KEY")
client = OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")

prompt = """
你是一个运行在受限工作区内的 AI 助手。
你的根目录被锁定在 ./workspace 下，你无法访问除此之外的任何路径。
1. **先观察后行动**：处理文件前，先调用 `ls` 查看工作区内容。
2. **工具链使用**：你可以连续调用工具，例如先 OCR 识别图片，再根据内容写文件。
3. **路径说明**：调用工具时请直接使用相对路径（如 'test.jpg' 而不是 './workspace/test.jpg'）。
"""

async def run_agent():
    # 配置服务器启动参数
    # 确保 server.py 就在同一目录下，如果不是，请写绝对路径
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    # 1. 建立标准输入输出连接
    async with stdio_client(server_params) as (read, write):
        # 2. 建立会话
        async with ClientSession(read, write) as session:
            # 3. 初始化连接
            await session.initialize()
            
            # 获取工具列表 (注意：session 返回的是对象列表)
            tools_data = await session.list_tools()
            
            # 将工具转换为 OpenAI 格式
            # 注意：底层库返回的属性名可能是 inputSchema 而非 input_schema
            openai_tools = []
            for t in tools_data.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema 
                    }
                })

            print("✅ Agent 已连接到 Server,请输入你的指令。")
            messages = [{"role": "system", "content": prompt}]

            while True:
                user_input = input("\n👤 用户: ")
                if user_input.lower() == 'exit': break
                
                messages.append({"role": "user", "content": user_input})

                while True:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto"
                    )

                    response_message = response.choices[0].message
                    if not response_message.tool_calls:
                        print(f"\n🤖 Agent: {response_message.content}")
                        messages.append(response_message)
                        break

                    messages.append(response_message)
                    
                    for tool_call in response_message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        print(f"🛠️  执行工具: {func_name}({func_args})")
                        
                        # 4. 调用工具
                        # 注意：底层调用方法是 session.call_tool
                        result = await session.call_tool(func_name, func_args)
                        
                        # 处理返回内容 (MCP 返回的是 content 列表)
                        result_text = "".join([c.text for c in result.content if hasattr(c, 'text')])
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text
                        })
                    
                    print("🧠 Agent 正在思考下一步...")

if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\n退出。")