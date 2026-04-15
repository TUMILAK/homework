import os
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

app = FastAPI()

gemini_client = genai.Client(api_key=os.getenv("GMN_API_KEY"))

ds_client = AsyncOpenAI(
    api_key=os.getenv("DPSK_API_KEY"), 
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """你是一个运行在受限工作区内的 AI 助手。根目录为 ./workspace。
1. 先观察后行动 (ls)。2. 支持工具链协同。3. 直接使用相对路径。"""

def clean_schema(schema):
    if not isinstance(schema, dict): return schema
    return {k: clean_schema(v) for k, v in schema.items() 
            if k not in ["additional_properties", "additionalProperties"]}

# --- HTML 模板 (新增了选择模型遮罩层) ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>MCP Multi-Model Workspace</title>
    <style>
        :root {
            --bg-color: #131314;
            --sidebar-color: #1e1f20;
            --text-color: #e3e3e3;
            --user-bubble: #2b2b2b;
            --accent-color: #4285f4;
        }
        body, html { height: 100%; margin: 0; font-family: sans-serif; background: var(--bg-color); color: var(--text-color); overflow: hidden; }
        
        /* 模型选择遮罩 */
        #model-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); display: flex; justify-content: center; align-items: center;
            z-index: 1000; flex-direction: column;
        }
        .model-card {
            background: #2b2b2b; padding: 30px; border-radius: 15px; width: 300px;
            text-align: center; cursor: pointer; margin: 10px; border: 2px solid transparent;
            transition: 0.3s;
        }
        .model-card:hover { border-color: var(--accent-color); background: #333; }
        .model-card h2 { margin: 0; color: var(--accent-color); }

        .container { display: flex; height: 100vh; filter: blur(10px); transition: 0.5s; }
        .sidebar { width: 260px; background: var(--sidebar-color); padding: 20px; border-right: 1px solid #333; }
        .main-content { flex: 1; display: flex; flex-direction: column; position: relative; }
        #chat-window { flex: 1; overflow-y: auto; padding: 40px 15%; scroll-behavior: smooth; }
        
        .message-row { margin-bottom: 25px; display: flex; flex-direction: column; }
        .user-row { align-items: flex-end; }
        .agent-row { align-items: flex-start; }
        .message { max-width: 85%; padding: 12px 16px; border-radius: 18px; line-height: 1.6; }
        .user-msg { background: var(--user-bubble); }
        .thought-msg { font-size: 13px; color: #aaa; background: #202123; border-radius: 12px; padding: 10px; border-left: 3px solid var(--accent-color); width: 90%; font-family: monospace; margin: 5px 0; }
        .result-msg { font-size: 12px; color: #888; background: #1a1a1c; padding: 8px; margin-top: 5px; border-radius: 4px; white-space: pre-wrap; }

        .input-area { position: absolute; bottom: 30px; left: 15%; right: 15%; background: #202124; border-radius: 28px; display: flex; padding: 10px 20px; }
        #input { flex: 1; background: transparent; border: none; color: white; padding: 10px; outline: none; }
        .send-btn { background: none; border: none; color: var(--accent-color); cursor: pointer; font-weight: bold; }
    </style>
</head>
<body>
    <div id="model-overlay">
        <h1 style="margin-bottom: 30px;">选择驱动模型</h1>
        <div style="display: flex;">
            <div class="model-card" onclick="selectModel('gemini')">
                <h2>Gemini 2.0</h2>
            </div>
            <div class="model-card" onclick="selectModel('deepseek')">
                <h2>DeepSeek</h2>
            </div>
        </div>
    </div>

    <div class="container" id="main-ui">
        <div class="sidebar">
            <h3 id="current-model-display">未连接</h3>
            <div style="font-size: 12px; color: #888;">📁 Workspace 被锁定</div>
        </div>
        <div class="main-content">
            <div id="chat-window"></div>
            <div class="input-area">
                <input type="text" id="input" placeholder="输入指令..." onkeydown="if(event.key==='Enter') send()">
                <button class="send-btn" onclick="send()">发送</button>
            </div>
        </div>
    </div>

    <script>
        var ws = new WebSocket("ws://" + window.location.host + "/ws");
        var chatWindow = document.getElementById('chat-window');
        var selectedModel = "";

        function selectModel(type) {
            selectedModel = type;
            ws.send(JSON.stringify({type: "init", model: type}));
            document.getElementById('model-overlay').style.display = 'none';
            document.getElementById('main-ui').style.filter = 'none';
            document.getElementById('current-model-display').innerText = "当前: " + type.toUpperCase();
            addMessage('模型已就绪');
        }

        ws.onmessage = function(event) {
            var data = JSON.parse(event.data);
            addMessage(data.role, data.content);
        };

        function addMessage(role, content) {
            var row = document.createElement('div');
            row.className = 'message-row ' + (role === 'User' ? 'user-row' : 'agent-row');
            var msgDiv = document.createElement('div');
            
            if (role === 'User') { msgDiv.className = 'message user-msg'; }
            else if (role.includes('工具') || role.includes('Thinking')) { msgDiv.className = 'thought-msg'; }
            else { msgDiv.className = 'message agent-msg'; }
            
            msgDiv.innerHTML = `<b>${role}:</b><br>${content}`;
            row.appendChild(msgDiv);
            chatWindow.appendChild(row);
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }

        function send() {
            var input = document.getElementById('input');
            if (!input.value.trim()) return;
            addMessage('User', input.value);
            ws.send(JSON.stringify({type: "message", content: input.value}));
            input.value = '';
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(HTML_TEMPLATE)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_data = await session.list_tools()
            
            # --- 转换两种格式的工具 ---
            function_declarations = []
            openai_tools = []
            for t in tools_data.tools:
                decl = {"name": t.name, "description": t.description, "parameters": clean_schema(t.inputSchema)}
                function_declarations.append(decl)
                openai_tools.append({"type": "function", "function": decl})
            
            gemini_tools = [{"function_declarations": function_declarations}] if function_declarations else []
            
            history = []
            current_model = None

            while True:
                data = await websocket.receive_json()
                
                # 处理初始选择
                if data.get("type") == "init":
                    current_model = data["model"]
                    if current_model == "deepseek":
                        history.append({"role": "system", "content": SYSTEM_PROMPT})
                    continue

                # 处理用户消息
                user_text = data["content"]
                if current_model == "deepseek":
                    history.append({"role": "user", "content": user_text})
                else:
                    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
                
                # --- 开始思考/工具循环 ---
                while True:
                    # 避免 429 配额错误
                    await asyncio.sleep(1) 
                    
                    try:
                        if current_model == "deepseek":
                            # --- DeepSeek 请求 ---
                            response = await ds_client.chat.completions.create(
                                model="deepseek-chat",
                                messages=history,
                                tools=openai_tools if openai_tools else None
                            )
                            msg = response.choices[0].message
                            history.append(msg)

                            if msg.content:
                                await websocket.send_json({"role": "🤖 DeepSeek", "content": msg.content})
                            
                            if not msg.tool_calls: break

                            for tc in msg.tool_calls:
                                await websocket.send_json({"role": "🧠 Thinking", "content": f"调用工具: {tc.function.name}"})
                                res = await session.call_tool(tc.function.name, json.loads(tc.function.arguments))
                                res_text = "".join([c.text for c in res.content if hasattr(c, 'text')])
                                
                                await websocket.send_json({"role": "🛠️ 工具结果", "content": f"<div class='result-msg'>{res_text[:500]}</div>"})
                                history.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": res_text})

                        else:
                            # --- Gemini 请求 ---
                            response = gemini_client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=history,
                                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=gemini_tools)
                            )
                            if not response.candidates: break
                            res_msg = response.candidates[0].content
                            history.append(res_msg)

                            if response.text:
                                await websocket.send_json({"role": "🤖 Gemini", "content": response.text})
                            
                            t_calls = [p.function_call for p in res_msg.parts if p.function_call]
                            if not t_calls: break

                            for fc in t_calls:
                                await websocket.send_json({"role": "🧠 Thinking", "content": f"调用工具: {fc.name}"})
                                res = await session.call_tool(fc.name, fc.args)
                                res_text = "".join([c.text for c in res.content if hasattr(c, 'text')])
                                
                                await websocket.send_json({"role": "🛠️ 工具结果", "content": f"<div class='result-msg'>{res_text[:500]}</div>"})
                                history.append(types.Content(role="tool", parts=[types.Part(function_response=types.FunctionResponse(name=fc.name, response={"result": res_text}))]))

                    except Exception as e:
                        await websocket.send_json({"role": "⚠️ 错误", "content": str(e)})
                        break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)