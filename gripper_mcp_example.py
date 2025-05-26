#!/usr/bin/env python3
"""
DaHuan 夹爪 MCP 服务器使用示例

这个示例展示了如何使用 MCP 客户端与夹爪服务器进行交互。
"""

import asyncio
import json
from typing import Dict, Any


class MockMCPClient:
    """模拟 MCP 客户端，用于演示如何调用夹爪服务器的工具"""
    
    def __init__(self):
        self.tools = {}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """调用 MCP 工具"""
        if arguments is None:
            arguments = {}
        
        print(f"调用工具: {tool_name}")
        print(f"参数: {json.dumps(arguments, ensure_ascii=False, indent=2)}")
        
        # 这里应该是实际的 MCP 调用
        # 为了演示，我们返回模拟结果
        return {"success": True, "message": f"模拟调用 {tool_name} 成功"}


async def gripper_demo():
    """夹爪控制演示"""
    client = MockMCPClient()
    
    print("=== DaHuan 夹爪 MCP 服务器演示 ===\n")
    
    # 1. 连接夹爪
    print("1. 连接夹爪设备")
    result = await client.call_tool("connect_gripper", {
        "com_port": "/dev/ttyUSB0"
    })
    print(f"结果: {result}\n")
    
    # 2. 初始化夹爪
    print("2. 初始化夹爪")
    result = await client.call_tool("initialize_gripper")
    print(f"结果: {result}\n")
    
    # 3. 快速设置夹爪
    print("3. 快速设置夹爪")
    result = await client.call_tool("gripper_quick_setup", {
        "force": 80
    })
    print(f"结果: {result}\n")
    
    # 4. 设置夹爪参数
    print("4. 设置夹爪参数")
    
    # 设置力值
    result = await client.call_tool("set_gripper_force", {
        "force": 75
    })
    print(f"设置力值结果: {result}")
    
    # 设置速度
    result = await client.call_tool("set_gripper_velocity", {
        "velocity": 500
    })
    print(f"设置速度结果: {result}\n")
    
    # 5. 夹爪动作演示
    print("5. 夹爪动作演示")
    
    # 打开夹爪
    result = await client.call_tool("open_gripper")
    print(f"打开夹爪结果: {result}")
    
    # 等待一段时间
    await asyncio.sleep(1)
    
    # 关闭夹爪
    result = await client.call_tool("close_gripper", {
        "position": 300
    })
    print(f"关闭夹爪结果: {result}")
    
    # 再次打开
    result = await client.call_tool("set_gripper_position", {
        "position": 800
    })
    print(f"设置位置结果: {result}\n")
    
    # 6. 旋转功能演示
    print("6. 旋转功能演示")
    
    # 设置旋转参数
    result = await client.call_tool("set_rotate_velocity", {
        "velocity": 50
    })
    print(f"设置旋转速度结果: {result}")
    
    result = await client.call_tool("set_rotate_force", {
        "force": 60
    })
    print(f"设置旋转力值结果: {result}")
    
    # 绝对旋转
    result = await client.call_tool("rotate_gripper_absolute", {
        "angle": 1000
    })
    print(f"绝对旋转结果: {result}")
    
    # 相对旋转
    result = await client.call_tool("rotate_gripper_relative", {
        "angle": 500
    })
    print(f"相对旋转结果: {result}")
    
    # 获取旋转角度
    result = await client.call_tool("get_rotate_angle")
    print(f"获取旋转角度结果: {result}\n")
    
    print("=== 演示完成 ===")


def print_available_tools():
    """打印可用的工具列表"""
    tools = [
        {
            "name": "connect_gripper",
            "description": "连接到夹爪设备",
            "parameters": ["com_port (可选)"]
        },
        {
            "name": "initialize_gripper", 
            "description": "初始化夹爪",
            "parameters": []
        },
        {
            "name": "set_gripper_force",
            "description": "设置夹爪力值",
            "parameters": ["force (0-100)"]
        },
        {
            "name": "set_gripper_position",
            "description": "设置夹爪位置", 
            "parameters": ["position (0-1000)"]
        },
        {
            "name": "set_gripper_velocity",
            "description": "设置夹爪速度",
            "parameters": ["velocity (0-1000)"]
        },
        {
            "name": "open_gripper",
            "description": "打开夹爪",
            "parameters": []
        },
        {
            "name": "close_gripper",
            "description": "关闭夹爪",
            "parameters": ["position (可选, 默认500)"]
        },
        {
            "name": "rotate_gripper_absolute",
            "description": "绝对旋转夹爪",
            "parameters": ["angle (-32768 到 32767)"]
        },
        {
            "name": "rotate_gripper_relative", 
            "description": "相对旋转夹爪",
            "parameters": ["angle (-32768 到 32767)"]
        },
        {
            "name": "set_rotate_velocity",
            "description": "设置旋转速度",
            "parameters": ["velocity (1-100)"]
        },
        {
            "name": "set_rotate_force",
            "description": "设置旋转力值",
            "parameters": ["force (20-100)"]
        },
        {
            "name": "get_rotate_angle",
            "description": "获取当前旋转角度",
            "parameters": []
        },
        {
            "name": "gripper_quick_setup",
            "description": "快速设置夹爪",
            "parameters": ["force (可选, 默认100)"]
        }
    ]
    
    print("=== 可用的夹爪控制工具 ===\n")
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   描述: {tool['description']}")
        if tool['parameters']:
            print(f"   参数: {', '.join(tool['parameters'])}")
        else:
            print(f"   参数: 无")
        print()


if __name__ == "__main__":
    print("DaHuan 夹爪 MCP 服务器使用指南\n")
    
    print("1. 启动 MCP 服务器:")
    print("   python gripper_mcp_server.py\n")
    
    print("2. 可用工具列表:")
    print_available_tools()
    
    print("3. 运行演示:")
    asyncio.run(gripper_demo()) 