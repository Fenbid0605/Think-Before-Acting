#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主服务器文件 - Diana机器人MCP服务器
整合所有模块并提供MCP接口
"""

from fastmcp import FastMCP, Context
from mcp_tools import MCPToolsManager

# 创建 MCP 服务器
mcp = FastMCP("Diana Robot MCP Server")

# 创建工具管理器实例
tools_manager = MCPToolsManager()

# 注册MCP工具
@mcp.tool()
async def connect_robot(robot_ip: str, ctx: Context):
    """连接到Diana机器人
    
    Args:
        robot_ip: 机器人的IP地址
    
    Returns:
        连接结果
    """
    return await tools_manager.connect_robot(robot_ip, ctx)

@mcp.tool()
async def configure_tactile_sensor(threshold: int, sensitivity: int, ctx: Context):
    """配置触觉传感器参数
    
    Args:
        threshold: 触觉阈值 (0-255)
        sensitivity: 触觉灵敏度 (1-20)
    
    Returns:
        配置结果
    """
    return await tools_manager.configure_tactile_sensor(threshold, sensitivity, ctx)

@mcp.tool()
async def init_devices(ctx: Context):
    """初始化所有设备，包括机械臂、夹爪、相机和触觉传感器"""
    return await tools_manager.init_devices(ctx)

@mcp.tool()
async def voice_control(ctx: Context):
    """启动语音控制模式，等待语音指令"""
    return await tools_manager.voice_control(ctx)

@mcp.tool()
async def start_object_detection(ctx: Context, target_object=None):
    """启动物体检测并控制机械臂抓取，集成触觉反馈调整
    
    Args:
        target_object: 目标物体类别名称，如果为None则检测所有物体
    """
    return await tools_manager.start_object_detection(ctx, target_object)

@mcp.tool()
async def move_arm_to_position(x: float, y: float, z: float, ctx: Context):
    """移动机械臂到指定位置
    
    Args:
        x: X坐标 (mm)
        y: Y坐标 (mm)
        z: Z坐标 (mm)
    
    Returns:
        移动结果
    """
    return await tools_manager.move_arm_to_position(x, y, z, ctx)

@mcp.tool()
async def control_gripper(position: int, ctx: Context):
    """控制夹爪开合
    
    Args:
        position: 夹爪位置 (0-1000，0为完全闭合，1000为完全打开)
    
    Returns:
        控制结果
    """
    return await tools_manager.control_gripper(position, ctx)

@mcp.tool()
async def reset_arm_position(ctx: Context):
    """将机械臂重置到初始位置"""
    return await tools_manager.reset_arm_position(ctx)

@mcp.tool()
async def auto_pick_and_place(ctx: Context):
    """自动执行抓取放置任务，基于语音指令确定抓取物体和放置位置"""
    return await tools_manager.auto_pick_and_place(ctx)

@mcp.tool()
async def close_all_devices(ctx: Context):
    """关闭所有设备"""
    return await tools_manager.close_all_devices(ctx)

@mcp.tool()
async def process_natural_language(text: str, ctx: Context):
    """处理自然语言指令，将其转换为机器人控制命令
    
    Args:
        text: 自然语言指令
    
    Returns:
        处理结果
    """
    return await tools_manager.process_natural_language(text, ctx)

@mcp.tool()
async def help(ctx: Context):
    """显示所有可用命令"""
    return await tools_manager.help(ctx)

if __name__ == "__main__":
    # 启动MCP服务器
    print("正在启动Diana机器人MCP服务器...")
    mcp.run() 