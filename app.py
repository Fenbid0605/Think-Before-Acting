import os
import json
import asyncio
import subprocess
import uuid
import configparser
import ssl
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 配置加载
class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        
        # 设置默认值
        self.api_key = ""
        self.base_url = "https://api.nimapi.cloud/v1/chat/completions"
        self.model = "microsoft/phi-3-vision-128k"
        self.temperature = 0.7
        self.max_tokens = 4096
        self.host = "0.0.0.0"
        self.port = 8000
        
        # 尝试加载配置文件
        self.load_config()
    
    def load_config(self):
        """从config.ini加载配置"""
        try:
            # 检查配置文件是否存在，如果不存在则创建默认配置
            if not os.path.exists('config.ini'):
                self.create_default_config()
            
            self.config.read('config.ini')
            
            # 读取API配置
            if 'API' in self.config:
                self.api_key = self.config.get('API', 'api_key', fallback=self.api_key)
                self.base_url = self.config.get('API', 'base_url', fallback=self.base_url)
            
            # 读取模型配置
            if 'MODEL' in self.config:
                self.model = self.config.get('MODEL', 'model', fallback=self.model)
                self.temperature = self.config.getfloat('MODEL', 'temperature', fallback=self.temperature)
                self.max_tokens = self.config.getint('MODEL', 'max_tokens', fallback=self.max_tokens)
            
            # 读取服务器配置
            if 'SERVER' in self.config:
                self.host = self.config.get('SERVER', 'host', fallback=self.host)
                self.port = self.config.getint('SERVER', 'port', fallback=self.port)
            
            # 环境变量优先级高于配置文件
            env_api_key = os.environ.get("NVIDIA_API_KEY")
            if env_api_key:
                self.api_key = env_api_key
        
        except Exception as e:
            print(f"加载配置文件出错: {str(e)}")
    
    def create_default_config(self):
        """创建默认配置文件"""
        self.config['API'] = {
            'api_key': '在此处填写您的NVIDIA API密钥',
            'base_url': self.base_url
        }
        self.config['MODEL'] = {
            'model': self.model,
            'temperature': str(self.temperature),
            'max_tokens': str(self.max_tokens)
        }
        self.config['SERVER'] = {
            'host': self.host,
            'port': str(self.port)
        }
        
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)

config = Config()

# FastAPI 应用
app = FastAPI(title="多模态AI交互服务器")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 模板
templates = Jinja2Templates(directory="templates")

# CORS设置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 保存会话历史
sessions = {}

# 模型
class ChatMessage(BaseModel):
    role: str
    content: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False

class ToolCallRequest(BaseModel):
    session_id: str
    tool: str
    parameters: Dict[str, Any]

# MCP工具管理器
class MCPToolManager:
    def __init__(self):
        self.realsense_process = None
        self.diana_process = None
        self.tools = {
            "realsense": {
                "initialize_camera": self._call_realsense_tool,
                "stop_camera": self._call_realsense_tool,
                "get_camera_frames": self._call_realsense_tool,
                "get_point_cloud": self._call_realsense_tool,
                "get_depth_data": self._call_realsense_tool,
                "get_object_distance": self._call_realsense_tool,
            },
            "diana_robot": {
                "get_joint_positions": self._call_diana_tool,
                "get_tcp_position": self._call_diana_tool,
                "get_robot_state": self._call_diana_tool,
                "move_joints": self._call_diana_tool,
                "move_linear": self._call_diana_tool,
                "move_tcp_direction": self._call_diana_tool,
                "rotate_tcp_direction": self._call_diana_tool,
                "stop_robot": self._call_diana_tool,
                "resume_robot": self._call_diana_tool,
                "enable_freedriving": self._call_diana_tool,
                "release_brake": self._call_diana_tool,
                "hold_brake": self._call_diana_tool,
            }
        }
    
    async def ensure_realsense_running(self):
        if self.realsense_process is None or self.realsense_process.poll() is not None:
            self.realsense_process = subprocess.Popen(
                ["python", "realsense_mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            # 等待服务器启动
            await asyncio.sleep(1)
    
    async def ensure_diana_running(self):
        if self.diana_process is None or self.diana_process.poll() is not None:
            self.diana_process = subprocess.Popen(
                ["python", "diana_mcp_server.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            # 等待服务器启动
            await asyncio.sleep(1)
    
    async def _call_realsense_tool(self, tool_name, parameters):
        await self.ensure_realsense_running()
        request = {
            "jsonrpc": "2.0",
            "method": tool_name,
            "params": parameters,
            "id": str(uuid.uuid4())
        }
        
        # 发送请求到子进程
        self.realsense_process.stdin.write(json.dumps(request) + "\n")
        self.realsense_process.stdin.flush()
        
        # 读取响应
        response_line = self.realsense_process.stdout.readline()
        try:
            response = json.loads(response_line)
            return response.get("result", {})
        except json.JSONDecodeError:
            return {"success": False, "message": "解析响应失败", "raw_response": response_line}
    
    async def _call_diana_tool(self, tool_name, parameters):
        await self.ensure_diana_running()
        request = {
            "jsonrpc": "2.0",
            "method": tool_name,
            "params": parameters,
            "id": str(uuid.uuid4())
        }
        
        # 发送请求到子进程
        self.diana_process.stdin.write(json.dumps(request) + "\n")
        self.diana_process.stdin.flush()
        
        # 读取响应，最多尝试3次，以避开可能的调试日志消息
        max_attempts = 3
        for attempt in range(max_attempts):
            response_line = self.diana_process.stdout.readline()
            
            # 尝试解析为JSON
            try:
                # 跳过明显的日志行
                if response_line.strip().startswith('['):
                    print(f"跳过日志行: {response_line.strip()}")
                    continue
                    
                response = json.loads(response_line)
                return response.get("result", {})
            except json.JSONDecodeError:
                # 最后一次尝试失败，返回错误
                if attempt == max_attempts - 1:
                    return {"success": False, "message": "解析响应失败", "raw_response": response_line}
                # 否则继续尝试下一行
                print(f"第{attempt+1}次解析响应失败，继续尝试...")
                continue
    
    async def call_tool(self, category, tool_name, parameters):
        if category == "realsense":
            return await self._call_realsense_tool(tool_name, parameters)
        elif category == "diana_robot":
            return await self._call_diana_tool(tool_name, parameters)
        else:
            return {"success": False, "message": f"未知工具类别: {category}"}

# 实例化工具管理器
tool_manager = MCPToolManager()

# 路由
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """返回主页"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "api_key_set": bool(config.api_key),
        "model": config.model,
        "temperature": config.temperature
    })

@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    return {
        "api_key_set": bool(config.api_key),
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens
    }

@app.post("/api/chat")
async def chat(request: Request):
    """处理聊天请求"""
    data = await request.json()
    session_id = data.get("session_id")
    message = data.get("message", "")
    image_data = data.get("image_data")
    
    if not session_id:
        session_id = str(uuid.uuid4())
        sessions[session_id] = []
    
    if session_id not in sessions:
        sessions[session_id] = []
    
    # 处理用户消息
    user_content = []
    
    # 添加文本内容
    if message:
        user_content.append({"type": "text", "text": message})
    
    # 添加图像内容（如果有）
    if image_data:
        # 去除Base64前缀
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}"
            }
        })
    
    # 保存原始格式的用户内容，用于会话历史记录
    original_user_content = user_content
    
    # 将多模态内容转换为纯文本格式，以适应phi-4-mini-instruct模型
    text_only_user_content = message
    if image_data:
        text_only_user_content += "\n[用户上传了一张图片]"
    
    # 构建消息历史 - 确保消息内容是字符串格式
    messages = []
    
    # 添加系统消息，告诉AI如何处理机器人指令
    system_message = (
        "你是一个多模态AI助手，可以控制戴安娜(Diana)机器人和RealSense相机。"
        "当用户请求控制机器人或获取相机数据时，直接回应你会执行该操作，无需解释你是AI或你无法物理交互。"
        "例如，当用户说'机器人向上抬0.1米'，你应该回应'好的，我会控制机器人向上移动0.1米'，而不是说'作为AI我无法控制物理设备'。"
        "机器人命令示例：向上移动、向前移动、旋转、获取位置、停止机器人。"
        "相机命令示例：拍照、获取深度图、获取点云数据。"
    )
    messages.append({"role": "system", "content": system_message})
    
    # 添加历史消息
    for m in sessions[session_id]:
        # 如果内容是列表，转换为字符串
        content = m["content"]
        if isinstance(content, list):
            # 提取所有文本内容
            text_parts = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append("[图片]")
            content = "\n".join(text_parts)
        messages.append({"role": m["role"], "content": content})
    
    # 添加当前用户消息，确保是字符串格式
    messages.append({"role": "user", "content": text_only_user_content})
    
    # 检查API密钥是否设置
    if not config.api_key or config.api_key == "在此处填写您的NVIDIA API密钥":
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "请在config.ini文件中设置NVIDIA API密钥"}
        )
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens
    }
    
    try:
        # 创建SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 使用自定义SSL上下文的连接器
        conn = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=conn) as session:
            async with session.post(config.base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return JSONResponse(
                        status_code=response.status,
                        content={"success": False, "error": f"API调用失败: {error_text}"}
                    )
                
                result = await response.json()
                
                # 处理返回结果
                assistant_message = result.get("choices", [{}])[0].get("message", {})
                
                # 更新会话历史
                sessions[session_id].append({"role": "user", "content": original_user_content})
                
                # 确保assistant_message的content是数组格式，与前端预期一致
                if isinstance(assistant_message.get("content"), str):
                    formatted_content = [{"type": "text", "text": assistant_message["content"]}]
                    assistant_message_for_history = {
                        "role": assistant_message.get("role", "assistant"),
                        "content": formatted_content
                    }
                else:
                    assistant_message_for_history = assistant_message
                
                sessions[session_id].append(assistant_message_for_history)
                
                # 解析并处理工具调用
                tool_calls = []
                content = assistant_message.get("content", "")
                
                # 将字符串内容转为文本，用于解析
                if isinstance(content, list):
                    content_text = ""
                    for item in content:
                        if item.get("type") == "text":
                            content_text += item.get("text", "") + " "
                else:
                    content_text = content
                
                # 更智能地解析工具调用
                # 1. 直接检测工具调用的关键词
                explicit_tool_keywords = ["我需要调用工具", "让我调用", "我将调用", "可以调用", "使用工具"]
                explicit_tool_call = any(keyword in content_text for keyword in explicit_tool_keywords)
                
                # 2. 检测机器人动作关键词
                robot_action_keywords = {
                    "diana_robot/move_tcp_direction": ["向上", "向下", "向左", "向右", "向前", "向后", "抬高", "降低", "上移", "下移", "左移", "右移"],
                    "diana_robot/move_linear": ["移动到", "移动至", "前往", "到达位置", "线性移动"],
                    "diana_robot/rotate_tcp_direction": ["旋转", "转动", "顺时针", "逆时针", "翻转"],
                    "diana_robot/get_tcp_position": ["位置", "当前位置", "坐标", "在哪", "在哪里"],
                    "diana_robot/get_joint_positions": ["关节位置", "关节角度", "关节状态"],
                    "diana_robot/get_robot_state": ["状态", "机器人状态", "工作状态"],
                    "diana_robot/stop_robot": ["停止", "暂停", "停下"],
                    "diana_robot/resume_robot": ["继续", "恢复", "重新开始"],
                    "diana_robot/release_brake": ["松开抱闸", "释放抱闸", "解除抱闸", "松开刹车", "释放刹车"],
                    "diana_robot/hold_brake": ["抱闸", "抱紧", "刹车", "固定关节", "锁定关节"],
                    "realsense/get_camera_frames": ["拍照", "照片", "图像", "画面", "相机画面"],
                    "realsense/get_depth_data": ["深度", "深度数据", "距离", "深度图"],
                    "realsense/get_point_cloud": ["点云", "3D点云", "三维点云"]
                }
                
                # 检查是否有机器人动作关键词
                detected_tools = []
                for tool_path, keywords in robot_action_keywords.items():
                    if any(keyword in content_text.lower() for keyword in keywords):
                        category, tool_name = tool_path.split("/")
                        detected_tools.append({"category": category, "tool": tool_name})
                
                # 3. 分析用户最近的消息是否包含机器人指令
                last_user_text = ""  # 确保变量始终定义
                if session_id in sessions and sessions[session_id]:
                    last_user_msgs = [m for m in sessions[session_id][-3:] if m.get("role") == "user"]
                    if last_user_msgs:
                        last_user_content = last_user_msgs[-1].get("content", [])
                        if isinstance(last_user_content, list):
                            for item in last_user_content:
                                if item.get("type") == "text":
                                    last_user_text += item.get("text", "") + " "
                        else:
                            last_user_text = str(last_user_content)
                        
                        # 检查用户消息中的机器人指令关键词
                        robot_command_patterns = [
                            "机器人", "戴安娜", "diana", "移动", "移到", "抬", "转", "旋转", 
                            "上移", "下移", "左移", "右移", "停止", "暂停", "继续",
                            "拍照", "拍张照", "照片", "深度图", "点云",
                            "松开抱闸", "释放抱闸", "解除抱闸", "松开刹车", "释放刹车", 
                            "抱闸", "抱紧", "刹车", "固定关节", "锁定关节"
                        ]
                        
                        if any(pattern in last_user_text.lower() for pattern in robot_command_patterns):
                            # 根据用户命令添加可能的工具调用
                            direction_words = ["上", "下", "左", "右", "前", "后"]
                            if any(word in last_user_text for word in direction_words):
                                if not any(t["tool"] == "move_tcp_direction" for t in detected_tools):
                                    detected_tools.append({"category": "diana_robot", "tool": "move_tcp_direction"})
                
                # 根据检测结果添加工具调用
                if explicit_tool_call:
                    # 如果AI明确表示要调用工具，检查它提到的具体工具
                    for category in ["realsense", "diana_robot"]:
                        for tool_name in tool_manager.tools.get(category, {}):
                            if tool_name.lower() in content_text.lower():
                                tool_calls.append({
                                    "category": category,
                                    "tool": tool_name,
                                    "parameters": {}  # 简化版本，参数需要通过前端输入
                                })
                
                # 添加通过关键词检测到的工具
                for tool in detected_tools:
                    if not any(t["category"] == tool["category"] and t["tool"] == tool["tool"] for t in tool_calls):
                        # 尝试提取参数
                        parameters = {}
                        
                        # 为不同工具提取不同参数
                        if tool["tool"] == "move_tcp_direction":
                            # 提取方向
                            direction = None
                            if "向上" in last_user_text or "上移" in last_user_text or "抬" in last_user_text:
                                direction = "z+"
                            elif "向下" in last_user_text or "下移" in last_user_text:
                                direction = "z-"
                            elif "向左" in last_user_text or "左移" in last_user_text:
                                direction = "y+"
                            elif "向右" in last_user_text or "右移" in last_user_text:
                                direction = "y-"
                            elif "向前" in last_user_text:
                                direction = "x+"
                            elif "向后" in last_user_text:
                                direction = "x-"
                            
                            # 提取距离
                            import re
                            distance_match = re.search(r'(\d+\.?\d*)(?:\s*)(米|m|毫米|mm|厘米|cm)', last_user_text)
                            distance = 0.1  # 默认值
                            if distance_match:
                                value = float(distance_match.group(1))
                                unit = distance_match.group(2)
                                # 转换为米
                                if unit in ["毫米", "mm"]:
                                    distance = value / 1000
                                elif unit in ["厘米", "cm"]:
                                    distance = value / 100
                                else:  # 米
                                    distance = value
                            
                            if direction:
                                parameters["direction"] = direction
                                parameters["distance"] = distance
                        
                        # 添加工具调用
                        tool_calls.append({
                            "category": tool["category"],
                            "tool": tool["tool"],
                            "parameters": parameters
                        })
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": {
                        "role": assistant_message.get("role", "assistant"),
                        "content": formatted_content if isinstance(assistant_message.get("content"), str) else assistant_message.get("content", "")
                    },
                    "tool_calls": tool_calls
                }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"请求处理错误: {str(e)}"}
        )

@app.post("/api/tool-call")
async def execute_tool_call(request: ToolCallRequest):
    """执行工具调用"""
    session_id = request.session_id
    tool_parts = request.tool.split("/")
    
    if len(tool_parts) != 2:
        return {"success": False, "error": "工具格式错误，应为 'category/tool_name'"}
    
    category, tool_name = tool_parts
    
    if session_id not in sessions:
        return {"success": False, "error": "会话不存在"}
    
    try:
        # 执行工具调用
        result = await tool_manager.call_tool(category, tool_name, request.parameters)
        
        # 将工具调用和结果添加到会话历史
        tool_call_message = {
            "role": "function",
            "name": f"{category}/{tool_name}",
            "content": json.dumps(result)
        }
        
        sessions[session_id].append(tool_call_message)
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {"success": False, "error": f"工具调用失败: {str(e)}"}

# 启动服务器
if __name__ == "__main__":
    # 确保目录存在
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    print(f"服务器配置:")
    print(f"- API密钥: {'已设置' if config.api_key and config.api_key != '在此处填写您的NVIDIA API密钥' else '未设置'}")
    print(f"- 模型: {config.model}")
    print(f"- 温度: {config.temperature}")
    print(f"- 最大令牌数: {config.max_tokens}")
    print(f"- 主机: {config.host}")
    print(f"- 端口: {config.port}")
    
    uvicorn.run("app:app", host=config.host, port=config.port, reload=True) 