#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 包含所有系统参数和配置
"""

import numpy as np

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
MODEL_PATH = "media/models/tiny.pt"      # Jetson推荐使用base或small
COMPUTE_TYPE = "int8"  # 根据CPU类型自动选择

# 关键词
TRIGGER_KEYWORDS = ["開始", "开始", "好了", "開", "始"]
OBJECT_TYPES = ["杯子", "水杯", "笔", "书", "手机", "玩具", "球", "盒子", "工具", "零件", "纸巾", "瓶子"]
PLACE_LOCATIONS = ["左边", "右边", "前面", "后面", "中间", "桌子", "工作台", "盒子里", "箱子", "白盒", "纸盒"]

# 机器人参数
DEFAULT_IP_ADDRESS = '192.168.10.75'
DEFAULT_VELOCITY = 0.2
DEFAULT_ACCELERATION = 0.2

# 触觉传感器参数
DEFAULT_TACTILE_THRESHOLD = 127  # 触觉阈值（0-255）
DEFAULT_TACTILE_SENSITIVITY = 5  # 触觉灵敏度
DEFAULT_GRIPPER_POSITION = 500  # 当前夹爪位置
GRIPPER_STEP = 50  # 夹爪调整步长

# 相机内参矩阵
CAMERA_MATRIX = np.array([
    [393.62592785,   0,         317.78003627],
    [  0,          396.64967581, 243.6218852 ],
    [  0,            0,           1        ]
])

# 手眼标定结果
HAND_EYE_ROTATION_MATRIX = np.array([
    [0.0396827,  -0.9950163,  -0.09147596],
    [-0.99920599, -0.03918962, -0.00718084],
    [ 0.00356014,  0.09168828, -0.99578139]
])

HAND_EYE_TRANSLATION_VECTOR = np.array([
    [125.63955172],
    [ -3.51564604],
    [ -34.3611117]
])

# 检测参数
MAX_DETECTION_COUNT = 30
MAX_RECOGNITION_COUNT = 15
MAX_TACTILE_ADJUSTMENTS = 5

# YOLO模型路径
YOLO_MODEL_PATH = 'weights/best_lab_yolo11x.pt' 