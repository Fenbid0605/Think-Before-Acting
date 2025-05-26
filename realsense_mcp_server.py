from fastmcp import FastMCP, Context
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import json
import base64
import sys
import os

# 创建 MCP 服务器
mcp = FastMCP("RealSense MCP 服务器")

# RealSense 管道和配置
pipeline = None
align = None

@mcp.tool()
async def initialize_camera(ctx: Context):
    """初始化 RealSense 相机
    
    Returns:
        初始化结果
    """
    global pipeline, align
    
    try:
        # 创建管道
        pipeline = rs.pipeline()
        
        # 创建配置
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 启动管道
        profile = pipeline.start(config)
        
        # 获取深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # 设置深度单位为毫米
        depth_scale = depth_sensor.get_depth_scale()
        
        # 创建对齐处理器（将深度帧对齐到彩色帧）
        align = rs.align(rs.stream.color)
        
        await ctx.info("RealSense 相机初始化成功")
        return {"success": True, "message": "相机初始化成功", "depth_scale": depth_scale}
    
    except Exception as e:
        await ctx.error(f"相机初始化失败: {str(e)}")
        return {"success": False, "message": f"相机初始化失败: {str(e)}"}

@mcp.tool()
async def stop_camera(ctx: Context):
    """停止 RealSense 相机
    
    Returns:
        停止结果
    """
    global pipeline
    
    try:
        if pipeline:
            pipeline.stop()
            pipeline = None
            await ctx.info("RealSense 相机已停止")
            return {"success": True, "message": "相机已停止"}
        else:
            return {"success": False, "message": "相机未初始化"}
    
    except Exception as e:
        return {"success": False, "message": f"停止相机失败: {str(e)}"}

@mcp.tool()
async def get_camera_frames(ctx: Context):
    """获取当前的彩色图像和深度图像
    
    Returns:
        Base64 编码的彩色图像和深度图像
    """
    global pipeline, align
    
    try:
        if not pipeline:
            return {"success": False, "message": "相机未初始化"}
        
        # 等待一组帧
        frames = pipeline.wait_for_frames()
        
        # 对齐帧
        aligned_frames = align.process(frames)
        
        # 获取对齐后的深度帧和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return {"success": False, "message": "无法获取有效帧"}
        
        # 将帧转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # 将深度图像映射到彩色可视化
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 将图像编码为 base64 字符串
        _, color_encoded = cv2.imencode('.jpg', color_image)
        _, depth_encoded = cv2.imencode('.jpg', depth_colormap)
        
        color_base64 = base64.b64encode(color_encoded).decode('utf-8')
        depth_base64 = base64.b64encode(depth_encoded).decode('utf-8')
        
        return {
            "success": True,
            "color_image": color_base64,
            "depth_image": depth_base64
        }
    
    except Exception as e:
        await ctx.error(f"获取帧失败: {str(e)}")
        return {"success": False, "message": f"获取帧失败: {str(e)}"}

@mcp.tool()
async def get_point_cloud(ctx: Context, sample_rate: int = 10):
    """获取点云数据
    
    Args:
        sample_rate: 采样率，每 sample_rate 个点取一个，降低数据量
    
    Returns:
        点云数据（坐标和颜色）
    """
    global pipeline, align
    
    try:
        if not pipeline:
            return {"success": False, "message": "相机未初始化"}
        
        # 等待一组帧
        frames = pipeline.wait_for_frames()
        
        # 对齐帧
        aligned_frames = align.process(frames)
        
        # 获取对齐后的深度帧和彩色帧
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return {"success": False, "message": "无法获取有效帧"}
        
        # 创建点云
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        
        # 获取顶点
        vertices = np.asanyarray(points.get_vertices())
        
        # 获取纹理坐标
        texture = np.asanyarray(points.get_texture_coordinates())
        
        # 获取颜色
        color_image = np.asanyarray(color_frame.get_data())
        
        # 降采样，减少数据量
        point_cloud_data = []
        
        for i in range(0, len(vertices), sample_rate):
            vertex = vertices[i]
            tex_coord = texture[i]
            
            # 确保纹理坐标在有效范围内
            x_tex = min(max(int(tex_coord.u * color_image.shape[1]), 0), color_image.shape[1] - 1)
            y_tex = min(max(int(tex_coord.v * color_image.shape[0]), 0), color_image.shape[0] - 1)
            
            color = color_image[y_tex, x_tex]
            
            point_cloud_data.append({
                "x": float(vertex.x),
                "y": float(vertex.y),
                "z": float(vertex.z),
                "r": int(color[2]),  # BGR 格式转 RGB
                "g": int(color[1]),
                "b": int(color[0])
            })
        
        return {
            "success": True,
            "point_cloud": point_cloud_data,
            "point_count": len(point_cloud_data)
        }
    
    except Exception as e:
        await ctx.error(f"获取点云失败: {str(e)}")
        return {"success": False, "message": f"获取点云失败: {str(e)}"}

@mcp.tool()
async def get_depth_data(ctx: Context):
    """获取原始深度数据
    
    Returns:
        深度数据矩阵（以毫米为单位）
    """
    global pipeline, align
    
    try:
        if not pipeline:
            return {"success": False, "message": "相机未初始化"}
        
        # 等待一组帧
        frames = pipeline.wait_for_frames()
        
        # 获取深度帧
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            return {"success": False, "message": "无法获取有效深度帧"}
        
        # 将深度帧转换为 numpy 数组
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 获取深度单位
        depth_sensor = frames.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        
        # 转换为毫米
        depth_mm = depth_image * depth_scale * 1000
        
        # 将深度数据转换为列表（降低采样率以减少数据量）
        depth_data = []
        step = 10  # 采样步长
        
        for y in range(0, depth_mm.shape[0], step):
            row = []
            for x in range(0, depth_mm.shape[1], step):
                row.append(float(depth_mm[y, x]))
            depth_data.append(row)
        
        return {
            "success": True,
            "depth_data": depth_data,
            "width": len(depth_data[0]),
            "height": len(depth_data),
            "original_width": depth_mm.shape[1],
            "original_height": depth_mm.shape[0],
            "sample_step": step
        }
    
    except Exception as e:
        await ctx.error(f"获取深度数据失败: {str(e)}")
        return {"success": False, "message": f"获取深度数据失败: {str(e)}"}

@mcp.tool()
async def get_object_distance(ctx: Context, x: int, y: int, radius: int = 5):
    """获取指定位置周围区域的平均距离
    
    Args:
        x: 图像中的 X 坐标
        y: 图像中的 Y 坐标
        radius: 计算平均距离的半径
    
    Returns:
        指定位置的距离（米）
    """
    global pipeline
    
    try:
        if not pipeline:
            return {"success": False, "message": "相机未初始化"}
        
        # 等待一组帧
        frames = pipeline.wait_for_frames()
        
        # 获取深度帧
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            return {"success": False, "message": "无法获取有效深度帧"}
        
        # 计算区域内的平均距离
        total_distance = 0.0
        valid_points = 0
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = x + dx
                ny = y + dy
                
                # 确保坐标在有效范围内
                if 0 <= nx < 640 and 0 <= ny < 480:
                    dist = depth_frame.get_distance(nx, ny)
                    if dist > 0:
                        total_distance += dist
                        valid_points += 1
        
        if valid_points == 0:
            return {"success": False, "message": "指定区域没有有效深度数据"}
        
        average_distance = total_distance / valid_points
        
        return {
            "success": True,
            "distance": average_distance,
            "valid_points": valid_points
        }
    
    except Exception as e:
        await ctx.error(f"获取距离失败: {str(e)}")
        return {"success": False, "message": f"获取距离失败: {str(e)}"}

# 运行服务器
if __name__ == "__main__":
    print("正在启动 RealSense MCP 服务器 (stdio 模式)...")
    
    # 运行服务器，使用 stdio 模式
    mcp.run(transport="stdio") 