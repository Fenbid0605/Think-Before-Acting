#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import queue
import wave
import pyaudio
import numpy as np
import subprocess
import re
import struct
import threading
import whisper
from queue import Queue
import pyttsx3
import json
import argparse
import asyncio  # 添加asyncio导入

import ControlGripper
from bin import DianaApi
from ControlRoot import ControlRoot
import PickAndPlace

import pyrealsense2 as rs
import cv2
import torch
import random
from logger import logger
from ultralytics import YOLO 
import math
import time
from scipy.spatial.transform import Rotation as R

from fastmcp import FastMCP, Context

# 创建 MCP 服务器
mcp = FastMCP("Diana Robot MCP Server")

# 全局变量
g_is_recording = False
g_audio_stream = None
g_pyaudio = None
transcriber = None

# 硬件参数配置
RECORD_CHANNEL = 1
RECORD_SAMPLE_RATE = 16000
CHUNK = 256
VOLUME_GAIN = 10.0  # XFM麦克风增益建议值
MIN_PROCESS_LENGTH = 16000  # 1秒触发处理16000 1.5秒建议24000 5秒建议80000 

# 模型参数配置
BEAM_SIZE = 3  # 3个束搜索  实时建议3-5
TEMPERATURE = 0  # 随机参数
LANGUAGE = "zh"  # 语言设置
NO_SPEECH_THRESHOLD = 0.8  # 静音过滤阈值

# 配置参数
model_path = "media/models/tiny.pt"      # Jetson推荐使用base或small
COMPUTE_TYPE = "int8" if 'intel' in os.popen('lscpu').read() else "float32"  # 根据CPU类型自动选择

# 关键词
TRIGGER_KEYWORDS = ["開始", "开始", "好了", "開", "始"]
OBJECT_TYPES = ["杯子", "水杯", "笔", "书", "手机", "玩具", "球", "盒子", "工具", "零件", "纸巾", "瓶子"]
PLACE_LOCATIONS = ["左边", "右边", "前面", "后面", "中间", "桌子", "工作台", "盒子里", "箱子", "白盒", "纸盒"]

# 全局对象
pipeline = None
depth_frame = None
color_image = None
gripper = None
camera_matrix = None
POSES = [0.0] * 6
hand_eye_matrix = None
ipAddress = '192.168.10.75'
vel = 0.2
acc = 0.2
tactile_camera = None  # 触觉传感器相机对象
tactile_image = None   # 触觉图像
tactile_threshold = 127  # 触觉阈值（0-255）
tactile_sensitivity = 5  # 触觉灵敏度
current_gripper_position = 500  # 当前夹爪位置
gripper_step = 50  # 夹爪调整步长


def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    random_color = f"#{r:02x}{g:02x}{b:02x}"
    return random_color
    
def xyz_rpy_to_homogeneous_matrix(x, y, z, roll, pitch, yaw):
    # 将Roll、Pitch、Yaw转换为旋转矩阵
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotation_matrix = r.as_matrix()

    # 创建4x4的齐次变换矩阵
    homogeneous_matrix = np.eye(4)

    # 将旋转矩阵放置在左上角的3x3部分
    homogeneous_matrix[:3, :3] = rotation_matrix

    # 将平移向量放置在第四列的前三个元素中
    homogeneous_matrix[:3, 3] = [x, y, z]

    return homogeneous_matrix

def object_distance_measure(left, right, top, bottom):
    global depth_frame, color_image
    width = right - left
    height = bottom - top
    # 测距的区域
    roi_lx = int(left + width/4)
    roi_rx = int(right - width/4)
    roi_ty = int(top + height/4)
    roi_by = int(bottom - height/4)

    center_x = int(left + width/2)
    center_y = int(top + height/2)
    cv2.circle(color_image, (center_x, center_y), 5, (0,0,255), 0)

    depth_points = []
    depth_means = []

    # 获取目标框内的物体距离，并进行均值滤波
    for j in range(50):
        rand_x = random.randint(roi_lx, roi_rx)
        rand_y = random.randint(roi_ty, roi_by)
        depth_point = round(depth_frame.get_distance(rand_x, rand_y)*100, 2)
        if depth_point != 0:
            depth_points.append(depth_point)
    depth_object = np.mean(depth_points)
    
    return center_x, center_y, depth_object

def Predict(model, img, classes=[], min_conf=0.5, device="cpu"):
    """
    Using Predict Model to predict objects in img.
    Input classes to choose which to output.
    eg. Predict(chosen_model, img_input, classes=[human], min_conf=0.5)
    """
    if classes:
        results = model.predict(
            img, classes=classes, conf=min_conf, device=device, stream=True
        )
    else:
        results = model.predict(img, conf=min_conf, device=device, stream=True)
    return results
 
def Predict_and_detect(model, img, classes=[], min_conf=0.5, rectangle_thickness=2, text_thickness=1, device="cpu"):
    global camera_matrix, POSES, ipAddress, vel, acc, depth_frame, hand_eye_matrix, gripper
    """
    Using Predict Model to predict objects in img and detect the objects out.
    Input classes to choose which to output.
    eg. Predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1)
    """
    results = Predict(model, img, classes, min_conf=min_conf, device=device)
    for result in results:
        for box in result.boxes:
            left, top, right, bottom = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )
            width = right - left
            height = bottom - top
            Obj_x = int(left + width/2)
            Obj_y = int(top + height/2)
            
            confidence = box.conf.tolist()[0]
            label = int(box.cls[0])
            color = 180
            caption = f"{result.names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]

            # 获取深度图像
            depth_image = np.asanyarray(depth_frame.get_data())

            # 获取目标点的深度值
            depth_value = depth_image[Obj_y, Obj_x]  # 单位mm

            # 将像素坐标转换为相机坐标
            pixel_coords = np.array([Obj_x, Obj_y, 1])
            inv_camera_matrix = np.linalg.inv(camera_matrix)
            normalized_coords = inv_camera_matrix @ pixel_coords
            camera_coords = depth_value * normalized_coords  # 得到相机系下的三位坐标

            camera_coords_homogeneous = np.append(camera_coords, 1)
            arm_coords_homogeneous = hand_eye_matrix @ camera_coords_homogeneous  # 转变到工具系
            arm_coords = arm_coords_homogeneous[:3]

            DianaApi.getTcpPos(POSES, ipAddress)
            angle_euler = POSES[3:]
            # 转化为欧拉角
            DianaApi.axis2RPY(angle_euler)
            # 转化为毫米
            for i in range(3):
                POSES[i] *= 1000.0
            for i in range(3):
                # 将旋转角度从弧度转换为角度
                angle_euler[i] = math.degrees(angle_euler[i])
            POSES[3:] = angle_euler
            arm_position = POSES
            # xyz_rpy转变为too12base的齐次变换矩阵
            tool_base_matrix = xyz_rpy_to_homogeneous_matrix(arm_position[0], arm_position[1], arm_position[2],
                                                            arm_position[3], arm_position[4], arm_position[5])

            base_coords_homogeneous = tool_base_matrix @ arm_coords_homogeneous  # 工具系转换为机械臂基座系
            base_coords = base_coords_homogeneous[:3]

            print(f"base coordinates: {base_coords}")

            poses = (base_coords[0], base_coords[1], base_coords[2], 0.0, 0.0, 0.0)
            print(f"poses coordinates: {poses}")
            DianaApi.moveJToPose(poses, vel, acc, ipAddress)
            time.sleep(1)
            
            if base_coords[2] <= 10:
                gripper.Position(300)
                time.sleep(2)

    return img, results

class SpeakTTS:
    def __init__(self):
        # 文本内容
        self.preset_texts = [
            "好的，现在开始执行"
        ]
        
        # 初始化语音引擎
        self.engine = pyttsx3.init(driverName='espeak')
        self._configure_engine()
        
    def _configure_engine(self):
        """硬件优化配置"""
        self.engine.setProperty('rate', 160)    # 适配Nano的合理语速  160
        self.engine.setProperty('volume', 0.7)  # 防止音频削波
        self._select_chinese_voice()

    def _select_chinese_voice(self):
        """自动选择中文语音包"""
        voices = self.engine.getProperty('voices')
        for v in voices:
            if any(key in v.id.lower() for key in ['zh', 'chinese']):
                self.engine.setProperty('voice', v.id)
                print(f"已选择语音: {v.name}")
                return
        print("警告：未找到中文语音，使用默认语音")

    def sequential_play(self, interval=2):
        """开始播放文本"""
        print("开始播报...")
        for idx, text in enumerate(self.preset_texts, 1):
            print(f"正在播报第{idx}条: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
            self.engine.stop()  # 重置引擎状态
            time.sleep(interval)  # 语句间隔时间
        print("所有内容播报完成")

    def say(self, text):
        """播放指定文本"""
        print(f"正在播报: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()  # 重置引擎状态

class AudioBuffer:
    def __init__(self, min_length=MIN_PROCESS_LENGTH):  # min_length秒触发处理
        self.buffer = np.zeros(min_length*2, dtype=np.float32)  # 双倍长度环形缓冲
        self.min_length = min_length
        self.pointer = 0
        self._last_update = time.time()
        
    def add_chunk(self, chunk):
        # 覆盖式写入
        self._last_update = time.time()  # 记录最后更新时间
        start = self.pointer
        end = self.pointer + len(chunk)
        if end > len(self.buffer):
            overflow = end - len(self.buffer)
            self.buffer[start:] = chunk[:-overflow]
            self.buffer[:overflow] = chunk[-overflow:]
            self.pointer = overflow
        else:
            self.buffer[start:end] = chunk
            self.pointer = end % len(self.buffer)
        return self.pointer >= self.min_length  # 触发条件
    
    def need_process(self):
        return self.pointer >= self.min_length  

class RealtimeTranscriber:
    def __init__(self):
        self.audio_buffer = AudioBuffer()  # 缓冲对象
        self.result_queue = Queue()  # 结果队列
        # 模型加载优化
        try:
            print(f"正在加载模型：{model_path}")
            self.model = whisper.load_model(
                model_path,
                device="cpu",  # 强制使用CPU
                in_memory=True   # 提升加载速度
            ).float()  # CPU必须使用float32
            self.model.eval()  # 启用评估模式减少内存占用            
            self.last_text = ""
            
            # 加载测试音频
            test_audio = whisper.load_audio("test.wav")
            trans_result = self.model.transcribe(
                test_audio,
                language=LANGUAGE,
                task="transcribe",
                beam_size=BEAM_SIZE,
                temperature=TEMPERATURE,
                no_speech_threshold=NO_SPEECH_THRESHOLD  # 增强静音过滤
            )  

            text_snippet = trans_result['text'][:20]
            print(f"测试转录结果：{text_snippet}...")

        except Exception as e:
            print(f"模型加载失败：{str(e)}")
        
    def transcribe_stream(self):
        global g_is_recording
        print(f"g_is_recording: {g_is_recording} ")
        """实时转录线程"""
        while g_is_recording:
            try:
                # 动态触发处理
                if self.audio_buffer.need_process():
                    print(f"[{time.strftime('%H:%M:%S')}] 开始处理音频段（缓冲区填充度：{self.audio_buffer.pointer/self.audio_buffer.min_length:.1%}）")
                    # 取最新1.5秒数据（带50%重叠）
                    segment = np.concatenate([
                        self.audio_buffer.buffer[self.audio_buffer.pointer:],
                        self.audio_buffer.buffer[:self.audio_buffer.pointer]
                    ])[-self.audio_buffer.min_length:]  # 取最新min_length秒

                    result = self.model.transcribe(
                        segment,
                        language=LANGUAGE,
                        fp16=False,  # CPU必须使用FP16
                        task="transcribe",  # 明确转录任务（非翻译）
                        beam_size=BEAM_SIZE,  # 降低计算量
                        temperature=TEMPERATURE,   # 随机参数
                        no_speech_threshold=NO_SPEECH_THRESHOLD,  # 增强静音过滤
                        compression_ratio_threshold=2.2  # 抑制重复文本
                    )
                    self.result_queue.put(result["text"]) 
                    print(f"[{time.strftime('%H:%M:%S')}] 转录完成：{result['text'][:20]}...")
            except TimeoutError:
                print("转录处理超时，跳过当前片段")

def find_xfm_device_alsa():
    """ALSA设备查找优化版"""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        xfm_match = re.search(r'card (\d+): XFMDPV0018', result.stdout)
        return int(xfm_match.group(1)) if xfm_match else None
    except Exception as e:
        print(f"ALSA设备查找失败: {e}")
        return None

def audio_callback(in_data, *args):
    # 直接写入缓冲区
    global transcriber

    # 应用增益并转换格式
    amplified = apply_volume_gain(in_data, VOLUME_GAIN)
    np_data = np.frombuffer(amplified, dtype=np.int16).astype(np.float32) / 32768.0
    transcriber.audio_buffer.add_chunk(np_data)  # 直接写入类成员

    return (in_data, pyaudio.paContinue)

def apply_volume_gain(audio_data, gain):
    """带削波保护的增益控制"""
    count = len(audio_data) // 2
    shorts = struct.unpack(f"{count}h", audio_data)
    amplified = np.clip(np.array(shorts) * gain, -32768, 32767).astype(np.int16)
    return struct.pack(f"{count}h", *amplified)

def apply_volume_gain(audio_data, gain):
    """带削波保护的增益控制"""
    count = len(audio_data) // 2
    shorts = struct.unpack(f"{count}h", audio_data)
    amplified = np.clip(np.array(shorts) * gain, -32768, 32767).astype(np.int16)
    return struct.pack(f"{count}h", *amplified)

def analyze_tactile_image(image):
    """分析触觉图像深浅并返回调整建议
    
    Args:
        image: 触觉传感器图像
        
    Returns:
        调整建议: 1-增大夹爪宽度, 0-保持不变, -1-减小夹爪宽度
    """
    if image is None:
        return 0
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
        
    # 计算平均灰度值
    avg_intensity = np.mean(gray_image)
     
    # 根据平均灰度值判断压力大小
    # 灰度值低（图像深）表示压力大
    if avg_intensity < tactile_threshold - tactile_sensitivity:
        return -1  # 减小夹爪宽度
    # 灰度值高（图像浅）表示压力小
    elif avg_intensity > tactile_threshold + tactile_sensitivity:
        return 1   # 增大夹爪宽度
    else:
        return 0   # 保持不变

def adjust_gripper_with_tactile_feedback(ctx=None):
    """根据触觉反馈调整夹爪宽度"""
    global tactile_image, gripper, current_gripper_position, gripper_step
    
    if tactile_image is None or gripper is None:
        return False
    
    # 分析触觉图像
    adjustment = analyze_tactile_image(tactile_image)
    
    # 根据调整建议修改夹爪宽度
    if adjustment != 0:
        # 计算新的夹爪位置
        new_position = current_gripper_position + (adjustment * gripper_step)
        # 确保在有效范围内（0-1000）
        new_position = max(0, min(1000, new_position))
        
        # 如果位置有实际变化
        if new_position != current_gripper_position:
            # 更新夹爪位置
            gripper.Position(new_position)
            current_gripper_position = new_position
            
            if ctx:
                asyncio.create_task(ctx.info(f"根据触觉反馈调整夹爪位置为: {new_position}"))
            else:
                print(f"根据触觉反馈调整夹爪位置为: {new_position}")
            
            # 等待夹爪动作完成
            time.sleep(0.5)
            return True
    
    return False

# MCP工具函数
@mcp.tool()
async def connect_robot(robot_ip: str, ctx: Context):
    """连接到Diana机器人
    
    Args:
        robot_ip: 机器人的IP地址
    
    Returns:
        连接结果
    """
    global ipAddress
    ipAddress = robot_ip
    
    await ctx.info(f"正在连接到机器人 {robot_ip}...")
    
    # 创建网络连接信息元组
    netInfo = (robot_ip, 0, 0, 0, 0, 0)
    
    # 连接机器人
    result = DianaApi.initSrv(netInfo)
    
    if result:
        # 获取机器人当前位姿
        global POSES
        DianaApi.getTcpPos(POSES, ipAddress)
        await ctx.info(f'机器人当前位姿: {POSES}')
        
        return {"success": True, "message": "已成功连接到机器人"}
    else:
        time.sleep(0.1)
        e = DianaApi.getLastError()  # 获取最近的错误代码
        e_info = DianaApi.formatError(e)  # 获取错误的描述信息
        error_message = f'连接失败,错误码为：{e},错误描述信息为：{e_info}'
        await ctx.error(error_message)
        return {"success": False, "message": error_message}

# 触觉传感器配置工具
@mcp.tool()
async def configure_tactile_sensor(threshold: int, sensitivity: int, ctx: Context):
    """配置触觉传感器参数
    
    Args:
        threshold: 触觉阈值 (0-255)
        sensitivity: 触觉灵敏度 (1-20)
    
    Returns:
        配置结果
    """
    global tactile_threshold, tactile_sensitivity
    
    # 确保参数在有效范围内
    threshold = max(0, min(255, threshold))
    sensitivity = max(1, min(20, sensitivity))
    
    await ctx.info(f"配置触觉传感器: 阈值={threshold}, 灵敏度={sensitivity}")
    
    tactile_threshold = threshold
    tactile_sensitivity = sensitivity
    
    return {
        "success": True, 
        "message": f"触觉传感器配置已更新: 阈值={tactile_threshold}, 灵敏度={tactile_sensitivity}"
    }


@mcp.tool()
async def init_devices(ctx: Context):
    """初始化所有设备，包括机械臂、夹爪、相机和触觉传感器"""
    global gripper, pipeline, camera_matrix, hand_eye_matrix, vel, acc, ipAddress, tactile_camera
    
    await ctx.info("正在初始化所有设备...")
    
    # 初始化机械臂
    await ctx.info("初始化机械臂...")
    vel = 0.2
    acc = 0.2
    DianaApi.getTcpPos(POSES, ipAddress)
    await ctx.info(f'机械臂当前位姿: {POSES}')
    
    # 初始化夹爪
    await ctx.info("初始化夹爪...")
    CR = ControlRoot()
    gripper = ControlGripper.SetCmd(CR)
    gripper.HandInit()
    gripper.Force(20)
    ControlGripper.InitGripper(gripper)
    
    # 初始化相机
    await ctx.info("初始化相机...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    
    # 初始化触觉传感器相机
    await ctx.info("初始化触觉传感器...")
    try:
        tactile_camera = cv2.VideoCapture(1)  # 尝试使用摄像头索引1
        if not tactile_camera.isOpened():
            tactile_camera = cv2.VideoCapture(2)  # 尝试索引2
            
        if tactile_camera.isOpened():
            await ctx.info("触觉传感器初始化成功")
        else:
            await ctx.warn("触觉传感器初始化失败，将在没有触觉反馈的情况下运行")
            tactile_camera = None
    except Exception as e:
        await ctx.warn(f"触觉传感器初始化错误: {str(e)}")
        tactile_camera = None
    
    # 创建对齐对象
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 相机内参矩阵
    camera_matrix = np.array([
        [393.62592785,   0,         317.78003627],
        [  0,          396.64967581, 243.6218852 ],
        [  0,            0,           1        ]
    ])
    
    # 手眼标定结果
    rotation_matrix = np.array([
        [0.0396827,  -0.9950163,  -0.09147596],
        [-0.99920599, -0.03918962, -0.00718084],
        [ 0.00356014,  0.09168828, -0.99578139]
    ])
    
    translation_vector = np.array([
        [125.63955172],
        [ -3.51564604],
        [ -34.3611117]
    ])
    
    hand_eye_matrix = np.eye(4)
    hand_eye_matrix[:3, :3] = rotation_matrix
    hand_eye_matrix[:3, 3] = translation_vector.flatten()
    
    # 初始化机械臂位置
    PickAndPlace.arm_pose_init(vel, acc, ipAddress)
    PickAndPlace.wait_move()
    
    # 初始化夹爪状态
    gripper.Position(1000)
    current_gripper_position = 1000
    time.sleep(2)
    
    return {"success": True, "message": "所有设备初始化完成"}

@mcp.tool()
async def voice_control(ctx: Context):
    """启动语音控制模式，等待语音指令"""
    global g_is_recording, g_pyaudio, g_audio_stream, transcriber
    
    await ctx.info("正在启动语音控制模式...")
    
    # 初始化硬件
    card_num = find_xfm_device_alsa()
    if card_num is None:
        await ctx.info("未找到XFM设备，使用默认麦克风")
        device_index = None
    else:
        device_index = int(card_num)
        await ctx.info(f"使用XFM设备 (卡号: {card_num})")
    
    # 初始化转录器
    transcriber = RealtimeTranscriber()
    
    # 初始化语音引擎
    Speaker = SpeakTTS()
    
    # 配置音频流参数
    g_pyaudio = pyaudio.PyAudio()
    g_audio_stream = g_pyaudio.open(
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
        g_audio_stream.start_stream()
        g_is_recording = True
        
        # 启动转录线程
        trans_thread = threading.Thread(target=transcriber.transcribe_stream)
        trans_thread.daemon = True
        trans_thread.start()
        
        await ctx.info("语音识别已启动，等待语音指令...")
        Speaker.say("语音识别已启动，请说出指令")
        
        # 识别计数器
        recognition_count = 0
        max_recognition_count = 15  # 增加最大识别次数，以捕获完整指令
        
        # 等待识别结果
        while g_is_recording and recognition_count < max_recognition_count:
            try:
                text = transcriber.result_queue.get(timeout=1.0)
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
                    Speaker.say(f"好的，我将抓取{command_details['object_type'] if command_details['object_type'] else '物体'}")
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
            Speaker.say(f"我将尝试抓取{command_details['object_type']}")
            return {
                "success": True,
                "message": "部分指令已识别",
                "details": command_details
            }
                
        if recognition_count >= max_recognition_count:
            await ctx.warn("超过最大识别次数，退出语音识别")
            return {"success": False, "message": "未识别到有效指令"}
            
    except Exception as e:
        await ctx.error(f"语音识别出错: {str(e)}")
        return {"success": False, "message": f"语音识别出错: {str(e)}"}
    finally:
        g_is_recording = False
        if g_audio_stream:
            g_audio_stream.stop_stream()
            g_audio_stream.close()
        if g_pyaudio:
            g_pyaudio.terminate()
        await ctx.info("语音识别已停止")

@mcp.tool()
async def start_object_detection(ctx: Context, target_object=None):
    """启动物体检测并控制机械臂抓取，集成触觉反馈调整
    
    Args:
        target_object: 目标物体类别名称，如果为None则检测所有物体
    """
    global pipeline, depth_frame, color_image, tactile_camera, tactile_image
    
    await ctx.info(f"正在启动物体检测...{f'目标物体: {target_object}' if target_object else '检测所有物体'}")
    
    # 初始化对齐对象
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 初始化模型
    model = YOLO(model=r'weights/best_lab_yolo11x.pt', task="detect")
    await ctx.info("已加载YOLOv11模型")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    await ctx.info(f"使用设备: {device}")
    
    detection_count = 0
    max_detection_count = 30  # 最多检测30帧
    detected_coords = None  # 用于存储检测到的物体坐标
    
    try:
        while detection_count < max_detection_count:
            # 获取帧
            frames = pipeline.wait_for_frames()
            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # 获取触觉传感器图像
            if tactile_camera is not None and tactile_camera.isOpened():
                ret, tactile_image = tactile_camera.read()
                if not ret:
                    tactile_image = None
            
            if not depth_frame or not color_frame:
                continue
                
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 检测物体
            classes = []  # 默认检测所有类别
            
            # 如果指定了目标物体，过滤类别
            if target_object:
                # 获取模型支持的所有类别
                all_classes = model.names
                target_class_id = None
                
                # 查找指定物体的类别ID
                for class_id, class_name in all_classes.items():
                    if target_object in class_name or class_name in target_object:
                        target_class_id = class_id
                        await ctx.info(f"找到匹配类别: {class_name} (ID: {class_id})")
                        break
                
                if target_class_id is not None:
                    classes = [target_class_id]
                    await ctx.info(f"将只检测类别ID: {classes}")
            
            result_img, results = Predict_and_detect(model, color_image, classes=classes, min_conf=0.5, device=device)
            
            # 检查是否有检测结果
            if results and any(len(result.boxes) > 0 for result in results):
                # 提取检测到的物体坐标
                for result in results:
                    for box in result.boxes:
                        left, top, right, bottom = (
                            int(box.xyxy[0][0]),
                            int(box.xyxy[0][1]),
                            int(box.xyxy[0][2]),
                            int(box.xyxy[0][3]),
                        )
                        center_x, center_y, depth_value = object_distance_measure(left, right, top, bottom)
                        label = int(box.cls[0])
                        class_name = result.names[label]
                        
                        # 如果指定了目标物体，确认是否匹配
                        if target_object and target_object not in class_name and class_name not in target_object:
                            continue
                            
                        await ctx.info(f"检测到物体: {class_name}, 位置: ({center_x}, {center_y}), 深度: {depth_value}cm")
                        detected_coords = (center_x, center_y, depth_value)
                        
                        # 显示检测结果
                        cv2.rectangle(color_image, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(color_image, f"{class_name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imshow("Detection", color_image)
                        cv2.waitKey(1)
                        
                        break
                    if detected_coords:
                        break
                
                await ctx.info("已检测到物体，开始抓取")
                
                # 抓取后使用触觉反馈调整夹爪
                await ctx.info("使用触觉反馈优化抓取...")
                
                # 循环调整夹爪直到获得适当的压力或达到最大尝试次数
                max_adjustments = 5
                adjustment_count = 0
                
                while adjustment_count < max_adjustments:
                    # 更新触觉传感器图像
                    if tactile_camera is not None and tactile_camera.isOpened():
                        ret, tactile_image = tactile_camera.read()
                        if not ret:
                            break
                    else:
                        break
                    
                    # 显示触觉图像
                    if tactile_image is not None:
                        cv2.imshow("Tactile Sensor", tactile_image)
                        cv2.waitKey(1)
                    
                    # 根据触觉反馈调整夹爪
                    if adjust_gripper_with_tactile_feedback(ctx):
                        adjustment_count += 1
                        await ctx.info(f"触觉调整 {adjustment_count}/{max_adjustments}")
                        time.sleep(0.5)
                    else:
                        await ctx.info("触觉反馈显示抓取力度适中，完成抓取")
                        break
                
                # 关闭显示窗口
                cv2.destroyAllWindows()
                
                return {
                    "success": True, 
                    "message": "已检测并抓取物体，触觉反馈优化完成",
                    "object_type": class_name if detected_coords else None,
                    "coordinates": detected_coords
                }
            
            detection_count += 1
            
        await ctx.warn("在最大检测次数内未检测到物体")
        # 关闭显示窗口
        cv2.destroyAllWindows()
        return {"success": False, "message": "未检测到物体"}
        
    except Exception as e:
        await ctx.error(f"物体检测出错: {str(e)}")
        cv2.destroyAllWindows()
        return {"success": False, "message": f"物体检测出错: {str(e)}"}


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
    global ipAddress, vel, acc
    
    await ctx.info(f"正在移动机械臂到位置 ({x}, {y}, {z})...")
    
    # 创建位姿数组
    poses = (x, y, z, 0.0, 0.0, 0.0)
    
    # 移动机械臂
    result = DianaApi.moveJToPose(poses, vel, acc, ipAddress)
    
    if result:
        await ctx.info("机械臂移动成功")
        return {"success": True, "message": "机械臂已移动到指定位置"}
    else:
        e = DianaApi.getLastError()
        e_info = DianaApi.formatError(e)
        error_message = f'机械臂移动失败,错误码为：{e},错误描述信息为：{e_info}'
        await ctx.error(error_message)
        return {"success": False, "message": error_message}

@mcp.tool()
async def control_gripper(position: int, ctx: Context):
    """控制夹爪开合
    
    Args:
        position: 夹爪位置 (0-1000，0为完全闭合，1000为完全打开)
    
    Returns:
        控制结果
    """
    global gripper
    
    position = max(0, min(1000, position))  # 确保在有效范围内
    
    await ctx.info(f"正在设置夹爪位置为 {position}...")
    
    try:
        gripper.Position(position)
        time.sleep(1)  # 等待夹爪动作完成
        await ctx.info("夹爪控制成功")
        return {"success": True, "message": f"夹爪已设置为位置 {position}"}
    except Exception as e:
        await ctx.error(f"夹爪控制失败: {str(e)}")
        return {"success": False, "message": f"夹爪控制失败: {str(e)}"}

@mcp.tool()
async def reset_arm_position(ctx: Context):
    """将机械臂重置到初始位置"""
    global ipAddress, vel, acc
    
    await ctx.info("正在重置机械臂位置...")
    
    try:
        PickAndPlace.arm_pose_init(vel, acc, ipAddress)
        PickAndPlace.wait_move()
        await ctx.info("机械臂已重置到初始位置")
        return {"success": True, "message": "机械臂已重置到初始位置"}
    except Exception as e:
        await ctx.error(f"重置机械臂位置失败: {str(e)}")
        return {"success": False, "message": f"重置机械臂位置失败: {str(e)}"}

@mcp.tool()
async def auto_pick_and_place(ctx: Context):
    """自动执行抓取放置任务，基于语音指令确定抓取物体和放置位置"""
    global gripper, current_gripper_position
    
    # 连接步骤
    await ctx.info("正在连接到机器人...")
    connect_result = await connect_robot("192.168.10.75", ctx)
    if not connect_result["success"]:
        return connect_result
    
    # 初始化设备
    await ctx.info("正在初始化设备...")
    init_result = await init_devices(ctx)
    if not init_result["success"]:
        return init_result
    
    # 打开夹爪准备抓取
    await ctx.info("打开夹爪准备抓取...")
    gripper.Position(1000)
    current_gripper_position = 1000
    time.sleep(1)
    
    # 语音控制
    await ctx.info("等待语音指令...")
    voice_result = await voice_control(ctx)
    if not voice_result["success"]:
        return voice_result
    
    # 获取语音指令详情
    command_details = voice_result.get("details", {})
    target_object = command_details.get("object_type")
    target_location = command_details.get("place_location")
    
    await ctx.info(f"语音指令解析结果: 目标物体={target_object}, 放置位置={target_location}")
    
    # 开始物体检测
    await ctx.info(f"开始物体检测和抓取...{'目标: ' + target_object if target_object else '检测任意物体'}")
    detection_result = await start_object_detection(ctx, target_object)
    
    if detection_result["success"]:
        # 确保夹爪抓取
        await ctx.info("确认抓取物体...")
        if current_gripper_position > 500:  # 如果夹爪位置仍然较大，可能未完全抓取
            gripper.Position(300)  # 设置较小的值确保抓紧
            current_gripper_position = 300
            time.sleep(1.5)
            await ctx.info("夹爪已关闭，物体已抓取")
        
        # 抬起物体
        await ctx.info("抬起物体...")
        current_pose = [0.0] * 6
        DianaApi.getTcpPos(current_pose, ipAddress)
        # 将位置坐标转换为毫米
        for i in range(3):
            current_pose[i] *= 1000.0
        # 抬高50mm
        lift_pose = (current_pose[0], current_pose[1], current_pose[2] + 50, current_pose[3], current_pose[4], current_pose[5])
        DianaApi.moveJToPose(lift_pose, vel, acc, ipAddress)
        time.sleep(2)
        
        # 根据语音指令确定放置位置
        place_pose = None
        if target_location:
            await ctx.info(f"根据语音指令放置到: {target_location}")
            # 根据位置关键词设置不同的放置坐标
            if "左边" in target_location:
                place_pose = (200, -150, 200, 0.0, 0.0, 0.0)
            elif "右边" in target_location:
                place_pose = (200, 150, 200, 0.0, 0.0, 0.0)
            elif "前面" in target_location:
                place_pose = (300, 0, 200, 0.0, 0.0, 0.0)
            elif "后面" in target_location:
                place_pose = (100, 0, 200, 0.0, 0.0, 0.0)
            elif "中间" in target_location:
                place_pose = (200, 0, 200, 0.0, 0.0, 0.0)
            elif "桌子" in target_location or "工作台" in target_location:
                place_pose = (250, 100, 200, 0.0, 0.0, 0.0)
            elif "盒子" in target_location or "箱子" in target_location:
                place_pose = (150, -100, 150, 0.0, 0.0, 0.0)
        
        if place_pose is None:
            # 默认放置位置
            place_pose = (300, 150, 200, 0.0, 0.0, 0.0)
            await ctx.info("使用默认放置位置")
        
        # 移动到放置位置
        await ctx.info(f"移动到放置位置: {place_pose}...")
        DianaApi.moveJToPose(place_pose, vel, acc, ipAddress)
        time.sleep(2)
        
        # 放下物体
        await ctx.info("放下物体...")
        gripper.Position(1000)
        current_gripper_position = 1000
        time.sleep(1.5)
        await ctx.info("物体已放置")
    
    # 重置机械臂位置
    await ctx.info("任务完成，重置机械臂位置...")
    await reset_arm_position(ctx)
    
    return {"success": True, "message": "抓取放置任务已完成"}

@mcp.tool()
async def close_all_devices(ctx: Context):
    """关闭所有设备"""
    global pipeline, ipAddress, tactile_camera
    
    await ctx.info("正在关闭所有设备...")
    
    # 关闭相机
    if pipeline:
        pipeline.stop()
        await ctx.info("相机已关闭")
    
    # 关闭触觉传感器
    if tactile_camera is not None and tactile_camera.isOpened():
        tactile_camera.release()
        cv2.destroyAllWindows()
        await ctx.info("触觉传感器已关闭")
    
    # 回到机械臂初始位置
    try:
        PickAndPlace.arm_pose_init(vel, acc, ipAddress)
        PickAndPlace.wait_move()
        await ctx.info("机械臂已重置到初始位置")
    except Exception as e:
        await ctx.warn(f"重置机械臂位置失败: {str(e)}")
    
    # 停止机械臂
    try:
        DianaApi.stop(ipAddress)
        DianaApi.destroySrv(ipAddress)
        await ctx.info("机械臂已停止")
    except Exception as e:
        await ctx.warn(f"停止机械臂失败: {str(e)}")
    
    return {"success": True, "message": "所有设备已关闭"}

@mcp.tool()
async def process_natural_language(text: str, ctx: Context):
    """处理自然语言指令，将其转换为机器人控制命令
    
    Args:
        text: 自然语言指令
    
    Returns:
        处理结果
    """
    await ctx.info(f"处理自然语言指令: {text}")
    
    # 连接相关指令
    if any(keyword in text for keyword in ["连接", "connect", "链接"]):
        return await connect_robot("192.168.10.75", ctx)
    
    # 初始化相关指令
    if any(keyword in text for keyword in ["初始化", "init", "setup", "设置"]):
        return await init_devices(ctx)
    
    # 语音控制相关指令
    if any(keyword in text for keyword in ["语音", "voice", "听", "说话"]):
        return await voice_control(ctx)
    
    # 物体检测相关指令
    if any(keyword in text for keyword in ["检测", "detect", "识别", "发现", "看"]):
        return await start_object_detection(ctx)
    
    # 夹爪控制相关指令
    if any(keyword in text for keyword in ["夹", "gripper", "抓", "放开"]):
        position = 300 if any(keyword in text for keyword in ["抓", "夹紧", "关闭"]) else 1000
        return await control_gripper(position, ctx)
    
    # 移动相关指令
    if any(keyword in text for keyword in ["移动", "move", "去", "到", "位置"]):
        # 默认位置
        return await move_arm_to_position(300, 0, 300, ctx)
    
    # 重置相关指令
    if any(keyword in text for keyword in ["重置", "reset", "回", "初始", "归位"]):
        return await reset_arm_position(ctx)
    
    # 自动任务相关指令
    if any(keyword in text for keyword in ["自动", "auto", "任务", "全部", "开始"]):
        return await auto_pick_and_place(ctx)
    
    # 关闭相关指令
    if any(keyword in text for keyword in ["关闭", "close", "停止", "结束"]):
        return await close_all_devices(ctx)
    
    await ctx.warn(f"无法理解指令: {text}")
    return {"success": False, "message": f"无法理解指令: {text}"}

@mcp.tool()
async def help(ctx: Context):
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

if __name__ == "__main__":
    # 启动MCP服务器
    print("正在启动Diana机器人MCP服务器...")
    mcp.run()



