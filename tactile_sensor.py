#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
触觉传感器模块 - 包含触觉图像分析和夹爪反馈调整功能
"""

import cv2
import numpy as np
import time

from config import *

class TactileSensor:
    def __init__(self, camera_index=1):
        self.camera = None
        self.camera_index = camera_index
        self.tactile_image = None
        self.threshold = DEFAULT_TACTILE_THRESHOLD
        self.sensitivity = DEFAULT_TACTILE_SENSITIVITY
        
    def initialize(self):
        """初始化触觉传感器相机"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                # 尝试其他索引
                for index in [2, 0, 3]:
                    self.camera = cv2.VideoCapture(index)
                    if self.camera.isOpened():
                        self.camera_index = index
                        break
                        
            if self.camera.isOpened():
                print(f"触觉传感器初始化成功，使用摄像头索引: {self.camera_index}")
                return True
            else:
                print("触觉传感器初始化失败")
                return False
                
        except Exception as e:
            print(f"触觉传感器初始化错误: {str(e)}")
            return False
            
    def capture_image(self):
        """捕获触觉图像"""
        if self.camera is None or not self.camera.isOpened():
            return False
            
        ret, self.tactile_image = self.camera.read()
        return ret
        
    def analyze_tactile_image(self, image=None):
        """分析触觉图像深浅并返回调整建议
        
        Args:
            image: 触觉传感器图像，如果为None则使用最新捕获的图像
            
        Returns:
            调整建议: 1-增大夹爪宽度, 0-保持不变, -1-减小夹爪宽度
        """
        if image is None:
            image = self.tactile_image
            
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
        if avg_intensity < self.threshold - self.sensitivity:
            return -1  # 减小夹爪宽度
        # 灰度值高（图像浅）表示压力小
        elif avg_intensity > self.threshold + self.sensitivity:
            return 1   # 增大夹爪宽度
        else:
            return 0   # 保持不变
            
    def get_pressure_level(self, image=None):
        """获取压力等级
        
        Returns:
            压力等级: "low", "medium", "high"
        """
        if image is None:
            image = self.tactile_image
            
        if image is None:
            return "unknown"
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            
        # 计算平均灰度值
        avg_intensity = np.mean(gray_image)
        
        if avg_intensity < self.threshold - self.sensitivity * 2:
            return "high"
        elif avg_intensity < self.threshold + self.sensitivity:
            return "medium"
        else:
            return "low"
            
    def configure(self, threshold, sensitivity):
        """配置触觉传感器参数"""
        self.threshold = max(0, min(255, threshold))
        self.sensitivity = max(1, min(20, sensitivity))
        return True
        
    def display_image(self, window_name="Tactile Sensor"):
        """显示触觉图像"""
        if self.tactile_image is not None:
            cv2.imshow(window_name, self.tactile_image)
            cv2.waitKey(1)
            
    def close(self):
        """关闭触觉传感器"""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()

class TactileFeedbackController:
    def __init__(self, tactile_sensor, robot_controller):
        self.tactile_sensor = tactile_sensor
        self.robot_controller = robot_controller
        self.gripper_step = GRIPPER_STEP
        
    def adjust_gripper_with_feedback(self, max_adjustments=MAX_TACTILE_ADJUSTMENTS):
        """根据触觉反馈调整夹爪宽度"""
        if not self.tactile_sensor.capture_image():
            print("无法获取触觉图像")
            return False
        
        adjustment_count = 0
        
        while adjustment_count < max_adjustments:
            # 更新触觉传感器图像
            if not self.tactile_sensor.capture_image():
                break
            
            # 显示触觉图像
            self.tactile_sensor.display_image()
            
            # 分析触觉图像
            adjustment = self.tactile_sensor.analyze_tactile_image()
            
            # 根据调整建议修改夹爪宽度
            if adjustment != 0:
                # 计算新的夹爪位置
                new_position = self.robot_controller.current_gripper_position + (adjustment * self.gripper_step)
                # 确保在有效范围内（0-1000）
                new_position = max(0, min(1000, new_position))
                
                # 如果位置有实际变化
                if new_position != self.robot_controller.current_gripper_position:
                    # 更新夹爪位置
                    if self.robot_controller.set_gripper_position(new_position):
                        print(f"根据触觉反馈调整夹爪位置为: {new_position}")
                        adjustment_count += 1
                        # 等待夹爪动作完成
                        time.sleep(0.5)
                    else:
                        break
                else:
                    break
            else:
                print("触觉反馈显示抓取力度适中，完成抓取")
                break
                
        return adjustment_count > 0
        
    def get_feedback_info(self):
        """获取触觉反馈信息"""
        if not self.tactile_sensor.capture_image():
            return None
            
        pressure_level = self.tactile_sensor.get_pressure_level()
        adjustment = self.tactile_sensor.analyze_tactile_image()
        
        return {
            "pressure_level": pressure_level,
            "adjustment_suggestion": adjustment,
            "current_gripper_position": self.robot_controller.current_gripper_position
        } 