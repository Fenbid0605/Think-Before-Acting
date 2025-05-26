#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP工具函数模块 - 包含所有MCP工具的实现
"""

import asyncio
import time
import queue
import threading
import pyaudio
import cv2
from fastmcp import Context

from config import *
from audio_processing import RealtimeTranscriber, SpeakTTS, find_xfm_device_alsa, create_audio_callback
from vision_processing import CameraManager, ObjectDetector
from robot_control import RobotController
from tactile_sensor import TactileSensor, TactileFeedbackController

class MCPToolsManager:
    def __init__(self):
        self.robot_controller = None
        self.camera_manager = None
        self.object_detector = None
        self.tactile_sensor = None
        self.tactile_feedback_controller = None
        self.transcriber = None
        self.speaker = None
        
        # 音频相关
        self.is_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        
    async def connect_robot(self, robot_ip: str, ctx: Context):
        """连接到Diana机器人"""
        await ctx.info(f"正在连接到机器人 {robot_ip}...")
        
        self.robot_controller = RobotController(robot_ip)
        
        if self.robot_controller.connect():
            await ctx.info(f'机器人当前位姿: {self.robot_controller.poses}')
            return {"success": True, "message": "已成功连接到机器人"}
        else:
            return {"success": False, "message": "连接机器人失败"}

    async def configure_tactile_sensor(self, threshold: int, sensitivity: int, ctx: Context):
        """配置触觉传感器参数"""
        threshold = max(0, min(255, threshold))
        sensitivity = max(1, min(20, sensitivity))
        
        await ctx.info(f"配置触觉传感器: 阈值={threshold}, 灵敏度={sensitivity}")
        
        if self.tactile_sensor is None:
            self.tactile_sensor = TactileSensor()
            
        self.tactile_sensor.configure(threshold, sensitivity)
        
        return {
            "success": True, 
            "message": f"触觉传感器配置已更新: 阈值={threshold}, 灵敏度={sensitivity}"
        }

    async def init_devices(self, ctx: Context):
        """初始化所有设备，包括机械臂、夹爪、相机和触觉传感器"""
        await ctx.info("正在初始化所有设备...")
        
        if self.robot_controller is None:
            await ctx.error("请先连接机器人")
            return {"success": False, "message": "请先连接机器人"}
        
        # 初始化机械臂
        await ctx.info("初始化机械臂...")
        self.robot_controller.get_current_pose()
        await ctx.info(f'机械臂当前位姿: {self.robot_controller.poses}')
        
        # 初始化夹爪
        await ctx.info("初始化夹爪...")
        if not self.robot_controller.initialize_gripper():
            await ctx.error("夹爪初始化失败")
            return {"success": False, "message": "夹爪初始化失败"}
        
        # 初始化相机
        await ctx.info("初始化相机...")
        self.camera_manager = CameraManager()
        if not self.camera_manager.initialize_camera():
            await ctx.error("相机初始化失败")
            return {"success": False, "message": "相机初始化失败"}
        
        # 初始化物体检测器
        await ctx.info("初始化物体检测器...")
        self.object_detector = ObjectDetector()
        
        # 初始化触觉传感器
        await ctx.info("初始化触觉传感器...")
        self.tactile_sensor = TactileSensor()
        if self.tactile_sensor.initialize():
            self.tactile_feedback_controller = TactileFeedbackController(
                self.tactile_sensor, self.robot_controller
            )
            await ctx.info("触觉传感器初始化成功")
        else:
            await ctx.warn("触觉传感器初始化失败，将在没有触觉反馈的情况下运行")
        
        # 初始化机械臂位置
        self.robot_controller.reset_to_initial_position()
        
        # 设置相机管理器的深度帧引用
        self.robot_controller.depth_frame = self.camera_manager.depth_frame
        
        return {"success": True, "message": "所有设备初始化完成"}

    async def voice_control(self, ctx: Context):
        """启动语音控制模式，等待语音指令"""
        await ctx.info("正在启动语音控制模式...")
        
        # 初始化硬件
        card_num = find_xfm_device_alsa()
        if card_num is None:
            await ctx.info("未找到XFM设备，使用默认麦克风")
            device_index = None
        else:
            device_index = int(card_num)
            await ctx.info(f"使用XFM设备 (卡号: {card_num})")
        
        # 初始化转录器和语音引擎
        self.transcriber = RealtimeTranscriber()
        self.speaker = SpeakTTS()
        
        # 配置音频流参数
        self.pyaudio_instance = pyaudio.PyAudio()
        audio_callback = create_audio_callback(self.transcriber)
        
        self.audio_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=RECORD_CHANNEL,
            rate=RECORD_SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            stream_callback=audio_callback,
            frames_per_buffer=CHUNK,
            start=False
        )
        
        # 用于存储命令详情的字典
        command_details = {
            "object_type": None,
            "place_location": None,
            "full_text": ""
        }
        
        try:
            self.audio_stream.start_stream()
            self.is_recording = True
            
            # 启动转录线程
            trans_thread = threading.Thread(
                target=self.transcriber.transcribe_stream,
                args=(lambda: self.is_recording,)
            )
            trans_thread.daemon = True
            trans_thread.start()
            
            await ctx.info("语音识别已启动，等待语音指令...")
            self.speaker.say("语音识别已启动，请说出指令")
            
            # 识别计数器
            recognition_count = 0
            
            # 等待识别结果
            while self.is_recording and recognition_count < MAX_RECOGNITION_COUNT:
                try:
                    text = self.transcriber.result_queue.get(timeout=1.0)
                    await ctx.info(f"识别结果: {text}")
                    command_details["full_text"] += text + " "
                    
                    # 检查是否包含触发词
                    has_trigger = any(keyword in text for keyword in TRIGGER_KEYWORDS)
                    
                    # 提取物体类型
                    for obj_type in OBJECT_TYPES:
                        if obj_type in text:
                            command_details["object_type"] = obj_type
                            await ctx.info(f"检测到物体: {obj_type}")
                    
                    # 提取放置位置
                    for location in PLACE_LOCATIONS:
                        if location in text:
                            command_details["place_location"] = location
                            await ctx.info(f"检测到位置: {location}")
                    
                    # 如果已经有足够的信息或者检测到触发词且有基本信息
                    if has_trigger and (command_details["object_type"] or recognition_count >= 3):
                        await ctx.info("检测到触发词，开始执行任务")
                        self.speaker.say(f"好的，我将抓取{command_details['object_type'] if command_details['object_type'] else '物体'}")
                        return {
                            "success": True, 
                            "message": "已识别触发指令", 
                            "details": command_details
                        }
                    
                    recognition_count += 1
                    
                except queue.Empty:
                    continue
                
            # 即使没有完整信息也尝试执行
            if command_details["full_text"]:
                if not command_details["object_type"]:
                    command_details["object_type"] = "物体"  # 默认值
                
                await ctx.info(f"未检测到完整指令，但将尝试执行: {command_details}")
                self.speaker.say(f"我将尝试抓取{command_details['object_type']}")
                return {
                    "success": True,
                    "message": "部分指令已识别",
                    "details": command_details
                }
                    
            if recognition_count >= MAX_RECOGNITION_COUNT:
                await ctx.warn("超过最大识别次数，退出语音识别")
                return {"success": False, "message": "未识别到有效指令"}
                
        except Exception as e:
            await ctx.error(f"语音识别出错: {str(e)}")
            return {"success": False, "message": f"语音识别出错: {str(e)}"}
        finally:
            self.is_recording = False
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            await ctx.info("语音识别已停止")

    async def start_object_detection(self, ctx: Context, target_object=None):
        """启动物体检测并控制机械臂抓取，集成触觉反馈调整"""
        await ctx.info(f"正在启动物体检测...{f'目标物体: {target_object}' if target_object else '检测所有物体'}")
        
        if self.camera_manager is None or self.object_detector is None:
            await ctx.error("相机或检测器未初始化")
            return {"success": False, "message": "相机或检测器未初始化"}
        
        detection_count = 0
        detected_coords = None
        detected_class_name = None
        
        try:
            while detection_count < MAX_DETECTION_COUNT:
                # 获取帧
                if not self.camera_manager.get_frames():
                    continue
                
                # 更新机器人控制器的深度帧引用
                self.robot_controller.depth_frame = self.camera_manager.depth_frame
                
                # 检测物体
                target_classes = [target_object] if target_object else None
                result_img, results = self.object_detector.detect_and_control(
                    self.camera_manager.color_image, 
                    self.robot_controller,
                    target_classes=target_classes,
                    min_conf=0.5
                )
                
                # 检查是否有检测结果
                if results and any(len(result.boxes) > 0 for result in results):
                    for result in results:
                        for box in result.boxes:
                            label = int(box.cls[0])
                            detected_class_name = result.names[label]
                            
                            # 如果指定了目标物体，确认是否匹配
                            if target_object and target_object not in detected_class_name and detected_class_name not in target_object:
                                continue
                            
                            await ctx.info(f"检测到物体: {detected_class_name}")
                            detected_coords = True
                            
                            # 显示检测结果
                            cv2.imshow("Detection", result_img)
                            cv2.waitKey(1)
                            
                            break
                        if detected_coords:
                            break
                    
                    await ctx.info("已检测到物体，开始抓取")
                    
                    # 抓取后使用触觉反馈调整夹爪
                    if self.tactile_feedback_controller:
                        await ctx.info("使用触觉反馈优化抓取...")
                        self.tactile_feedback_controller.adjust_gripper_with_feedback()
                        await ctx.info("触觉反馈优化完成")
                    
                    # 关闭显示窗口
                    cv2.destroyAllWindows()
                    
                    return {
                        "success": True, 
                        "message": "已检测并抓取物体，触觉反馈优化完成",
                        "object_type": detected_class_name,
                        "coordinates": detected_coords
                    }
                
                detection_count += 1
                
            await ctx.warn("在最大检测次数内未检测到物体")
            cv2.destroyAllWindows()
            return {"success": False, "message": "未检测到物体"}
            
        except Exception as e:
            await ctx.error(f"物体检测出错: {str(e)}")
            cv2.destroyAllWindows()
            return {"success": False, "message": f"物体检测出错: {str(e)}"}

    async def move_arm_to_position(self, x: float, y: float, z: float, ctx: Context):
        """移动机械臂到指定位置"""
        await ctx.info(f"正在移动机械臂到位置 ({x}, {y}, {z})...")
        
        if self.robot_controller is None:
            await ctx.error("机器人未连接")
            return {"success": False, "message": "机器人未连接"}
        
        poses = (x, y, z, 0.0, 0.0, 0.0)
        
        if self.robot_controller.move_to_pose(poses):
            await ctx.info("机械臂移动成功")
            return {"success": True, "message": "机械臂已移动到指定位置"}
        else:
            await ctx.error("机械臂移动失败")
            return {"success": False, "message": "机械臂移动失败"}

    async def control_gripper(self, position: int, ctx: Context):
        """控制夹爪开合"""
        position = max(0, min(1000, position))
        await ctx.info(f"正在设置夹爪位置为 {position}...")
        
        if self.robot_controller is None:
            await ctx.error("机器人未连接")
            return {"success": False, "message": "机器人未连接"}
        
        if self.robot_controller.set_gripper_position(position):
            time.sleep(1)
            await ctx.info("夹爪控制成功")
            return {"success": True, "message": f"夹爪已设置为位置 {position}"}
        else:
            await ctx.error("夹爪控制失败")
            return {"success": False, "message": "夹爪控制失败"}

    async def reset_arm_position(self, ctx: Context):
        """将机械臂重置到初始位置"""
        await ctx.info("正在重置机械臂位置...")
        
        if self.robot_controller is None:
            await ctx.error("机器人未连接")
            return {"success": False, "message": "机器人未连接"}
        
        if self.robot_controller.reset_to_initial_position():
            await ctx.info("机械臂已重置到初始位置")
            return {"success": True, "message": "机械臂已重置到初始位置"}
        else:
            await ctx.error("重置机械臂位置失败")
            return {"success": False, "message": "重置机械臂位置失败"}

    async def auto_pick_and_place(self, ctx: Context):
        """自动执行抓取放置任务，基于语音指令确定抓取物体和放置位置"""
        # 连接步骤
        await ctx.info("正在连接到机器人...")
        connect_result = await self.connect_robot(DEFAULT_IP_ADDRESS, ctx)
        if not connect_result["success"]:
            return connect_result
        
        # 初始化设备
        await ctx.info("正在初始化设备...")
        init_result = await self.init_devices(ctx)
        if not init_result["success"]:
            return init_result
        
        # 打开夹爪准备抓取
        await ctx.info("打开夹爪准备抓取...")
        self.robot_controller.set_gripper_position(1000)
        time.sleep(1)
        
        # 语音控制
        await ctx.info("等待语音指令...")
        voice_result = await self.voice_control(ctx)
        if not voice_result["success"]:
            return voice_result
        
        # 获取语音指令详情
        command_details = voice_result.get("details", {})
        target_object = command_details.get("object_type")
        target_location = command_details.get("place_location")
        
        await ctx.info(f"语音指令解析结果: 目标物体={target_object}, 放置位置={target_location}")
        
        # 开始物体检测
        await ctx.info(f"开始物体检测和抓取...{'目标: ' + target_object if target_object else '检测任意物体'}")
        detection_result = await self.start_object_detection(ctx, target_object)
        
        if detection_result["success"]:
            # 执行抓取和放置
            await ctx.info("执行抓取和放置...")
            if self.robot_controller.execute_pick_and_place(target_location):
                await ctx.info("抓取放置完成")
            else:
                await ctx.warn("抓取放置执行失败")
        
        # 重置机械臂位置
        await ctx.info("任务完成，重置机械臂位置...")
        await self.reset_arm_position(ctx)
        
        return {"success": True, "message": "抓取放置任务已完成"}

    async def close_all_devices(self, ctx: Context):
        """关闭所有设备"""
        await ctx.info("正在关闭所有设备...")
        
        # 关闭相机
        if self.camera_manager:
            self.camera_manager.stop_camera()
            await ctx.info("相机已关闭")
        
        # 关闭触觉传感器
        if self.tactile_sensor:
            self.tactile_sensor.close()
            await ctx.info("触觉传感器已关闭")
        
        # 重置机械臂并停止
        if self.robot_controller:
            try:
                self.robot_controller.reset_to_initial_position()
                await ctx.info("机械臂已重置到初始位置")
            except Exception as e:
                await ctx.warn(f"重置机械臂位置失败: {str(e)}")
            
            try:
                self.robot_controller.stop_robot()
                await ctx.info("机械臂已停止")
            except Exception as e:
                await ctx.warn(f"停止机械臂失败: {str(e)}")
        
        return {"success": True, "message": "所有设备已关闭"}

    async def process_natural_language(self, text: str, ctx: Context):
        """处理自然语言指令，将其转换为机器人控制命令"""
        await ctx.info(f"处理自然语言指令: {text}")
        
        # 连接相关指令
        if any(keyword in text for keyword in ["连接", "connect", "链接"]):
            return await self.connect_robot(DEFAULT_IP_ADDRESS, ctx)
        
        # 初始化相关指令
        if any(keyword in text for keyword in ["初始化", "init", "setup", "设置"]):
            return await self.init_devices(ctx)
        
        # 语音控制相关指令
        if any(keyword in text for keyword in ["语音", "voice", "听", "说话"]):
            return await self.voice_control(ctx)
        
        # 物体检测相关指令
        if any(keyword in text for keyword in ["检测", "detect", "识别", "发现", "看"]):
            return await self.start_object_detection(ctx)
        
        # 夹爪控制相关指令
        if any(keyword in text for keyword in ["夹", "gripper", "抓", "放开"]):
            position = 300 if any(keyword in text for keyword in ["抓", "夹紧", "关闭"]) else 1000
            return await self.control_gripper(position, ctx)
        
        # 移动相关指令
        if any(keyword in text for keyword in ["移动", "move", "去", "到", "位置"]):
            return await self.move_arm_to_position(300, 0, 300, ctx)
        
        # 重置相关指令
        if any(keyword in text for keyword in ["重置", "reset", "回", "初始", "归位"]):
            return await self.reset_arm_position(ctx)
        
        # 自动任务相关指令
        if any(keyword in text for keyword in ["自动", "auto", "任务", "全部", "开始"]):
            return await self.auto_pick_and_place(ctx)
        
        # 关闭相关指令
        if any(keyword in text for keyword in ["关闭", "close", "停止", "结束"]):
            return await self.close_all_devices(ctx)
        
        await ctx.warn(f"无法理解指令: {text}")
        return {"success": False, "message": f"无法理解指令: {text}"}

    async def help(self, ctx: Context):
        """显示所有可用命令"""
        commands = [
            {"名称": "connect_robot", "描述": "连接到Diana机器人"},
            {"名称": "init_devices", "描述": "初始化所有设备，包括机械臂、夹爪和相机"},
            {"名称": "voice_control", "描述": "启动语音控制模式，等待语音指令"},
            {"名称": "start_object_detection", "描述": "启动物体检测并控制机械臂抓取"},
            {"名称": "move_arm_to_position", "描述": "移动机械臂到指定位置"},
            {"名称": "control_gripper", "描述": "控制夹爪开合"},
            {"名称": "reset_arm_position", "描述": "将机械臂重置到初始位置"},
            {"名称": "auto_pick_and_place", "描述": "自动执行抓取放置任务"},
            {"名称": "close_all_devices", "描述": "关闭所有设备"},
            {"名称": "process_natural_language", "描述": "处理自然语言指令"}
        ]
        
        await ctx.info("可用命令列表:")
        for cmd in commands:
            await ctx.info(f"- {cmd['名称']}: {cmd['描述']}")
        
        return {"commands": commands} 