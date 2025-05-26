#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉处理模块 - 包含物体检测、深度测量和相机相关功能
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import random
import math
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

from config import *

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

def object_distance_measure(left, right, top, bottom, depth_frame, color_image):
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

    # 获取目标框内的物体距离，并进行均值滤波
    for j in range(50):
        rand_x = random.randint(roi_lx, roi_rx)
        rand_y = random.randint(roi_ty, roi_by)
        depth_point = round(depth_frame.get_distance(rand_x, rand_y)*100, 2)
        if depth_point != 0:
            depth_points.append(depth_point)
    depth_object = np.mean(depth_points)
    
    return center_x, center_y, depth_object

def predict_objects(model, img, classes=[], min_conf=0.5, device="cpu"):
    """
    Using Predict Model to predict objects in img.
    Input classes to choose which to output.
    """
    if classes:
        results = model.predict(
            img, classes=classes, conf=min_conf, device=device, stream=True
        )
    else:
        results = model.predict(img, conf=min_conf, device=device, stream=True)
    return results

def predict_and_detect(model, img, robot_controller, classes=[], min_conf=0.5, rectangle_thickness=2, text_thickness=1, device="cpu"):
    """
    Using Predict Model to predict objects in img and detect the objects out.
    """
    results = predict_objects(model, img, classes, min_conf=min_conf, device=device)
    
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
            depth_image = np.asanyarray(robot_controller.depth_frame.get_data())

            # 获取目标点的深度值
            depth_value = depth_image[Obj_y, Obj_x]  # 单位mm

            # 将像素坐标转换为相机坐标
            pixel_coords = np.array([Obj_x, Obj_y, 1])
            inv_camera_matrix = np.linalg.inv(robot_controller.camera_matrix)
            normalized_coords = inv_camera_matrix @ pixel_coords
            camera_coords = depth_value * normalized_coords  # 得到相机系下的三位坐标

            camera_coords_homogeneous = np.append(camera_coords, 1)
            arm_coords_homogeneous = robot_controller.hand_eye_matrix @ camera_coords_homogeneous  # 转变到工具系
            arm_coords = arm_coords_homogeneous[:3]

            # 获取当前机械臂位姿
            robot_controller.get_current_pose()
            arm_position = robot_controller.poses
            
            # xyz_rpy转变为tool2base的齐次变换矩阵
            tool_base_matrix = xyz_rpy_to_homogeneous_matrix(
                arm_position[0], arm_position[1], arm_position[2],
                arm_position[3], arm_position[4], arm_position[5]
            )

            base_coords_homogeneous = tool_base_matrix @ arm_coords_homogeneous  # 工具系转换为机械臂基座系
            base_coords = base_coords_homogeneous[:3]

            print(f"base coordinates: {base_coords}")

            poses = (base_coords[0], base_coords[1], base_coords[2], 0.0, 0.0, 0.0)
            print(f"poses coordinates: {poses}")
            
            # 移动机械臂到目标位置
            robot_controller.move_to_pose(poses)
            time.sleep(1)
            
            if base_coords[2] <= 10:
                robot_controller.set_gripper_position(300)
                time.sleep(2)

    return img, results

class CameraManager:
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.depth_frame = None
        self.color_frame = None
        self.color_image = None
        
    def initialize_camera(self):
        """初始化RealSense相机"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        
        # 创建对齐对象
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        return True
        
    def get_frames(self):
        """获取相机帧"""
        if not self.pipeline:
            return False
            
        frames = self.pipeline.wait_for_frames()
        # 对齐深度帧到颜色帧
        aligned_frames = self.align.process(frames)
        self.depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()
        
        if not self.depth_frame or not self.color_frame:
            return False
            
        # 转换为numpy数组
        self.color_image = np.asanyarray(self.color_frame.get_data())
        
        return True
        
    def stop_camera(self):
        """停止相机"""
        if self.pipeline:
            self.pipeline.stop()

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model = YOLO(model=model_path, task="detect")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def detect_objects(self, image, target_classes=None, min_conf=0.5):
        """检测图像中的物体"""
        classes = []
        
        # 如果指定了目标物体，过滤类别
        if target_classes:
            all_classes = self.model.names
            for target in target_classes:
                for class_id, class_name in all_classes.items():
                    if target in class_name or class_name in target:
                        classes.append(class_id)
                        print(f"找到匹配类别: {class_name} (ID: {class_id})")
                        break
        
        return predict_objects(self.model, image, classes=classes, min_conf=min_conf, device=self.device)
        
    def detect_and_control(self, image, robot_controller, target_classes=None, min_conf=0.5):
        """检测物体并控制机械臂"""
        classes = []
        
        if target_classes:
            all_classes = self.model.names
            for target in target_classes:
                for class_id, class_name in all_classes.items():
                    if target in class_name or class_name in target:
                        classes.append(class_id)
                        break
        
        return predict_and_detect(
            self.model, image, robot_controller, 
            classes=classes, min_conf=min_conf, device=self.device
        ) 