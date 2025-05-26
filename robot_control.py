#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人控制模块 - 包含机械臂控制、夹爪控制和位姿管理功能
"""

import time
import math
import numpy as np
from bin import DianaApi
import ControlGripper
from ControlRoot import ControlRoot
import PickAndPlace

from config import *

class RobotController:
    def __init__(self, ip_address=DEFAULT_IP_ADDRESS):
        self.ip_address = ip_address
        self.velocity = DEFAULT_VELOCITY
        self.acceleration = DEFAULT_ACCELERATION
        self.poses = [0.0] * 6
        self.gripper = None
        self.camera_matrix = CAMERA_MATRIX
        self.hand_eye_matrix = self._create_hand_eye_matrix()
        self.current_gripper_position = DEFAULT_GRIPPER_POSITION
        self.depth_frame = None  # 将由相机管理器设置
        
    def _create_hand_eye_matrix(self):
        """创建手眼标定矩阵"""
        hand_eye_matrix = np.eye(4)
        hand_eye_matrix[:3, :3] = HAND_EYE_ROTATION_MATRIX
        hand_eye_matrix[:3, 3] = HAND_EYE_TRANSLATION_VECTOR.flatten()
        return hand_eye_matrix
        
    def connect(self):
        """连接到机器人"""
        # 创建网络连接信息元组
        net_info = (self.ip_address, 0, 0, 0, 0, 0)
        
        # 连接机器人
        result = DianaApi.initSrv(net_info)
        
        if result:
            # 获取机器人当前位姿
            self.get_current_pose()
            print(f'机器人当前位姿: {self.poses}')
            return True
        else:
            time.sleep(0.1)
            e = DianaApi.getLastError()  # 获取最近的错误代码
            e_info = DianaApi.formatError(e)  # 获取错误的描述信息
            error_message = f'连接失败,错误码为：{e},错误描述信息为：{e_info}'
            print(error_message)
            return False
            
    def initialize_gripper(self):
        """初始化夹爪"""
        try:
            CR = ControlRoot()
            self.gripper = ControlGripper.SetCmd(CR)
            self.gripper.HandInit()
            self.gripper.Force(20)
            ControlGripper.InitGripper(self.gripper)
            
            # 初始化夹爪状态
            self.gripper.Position(1000)
            self.current_gripper_position = 1000
            time.sleep(2)
            return True
        except Exception as e:
            print(f"夹爪初始化失败: {str(e)}")
            return False
            
    def get_current_pose(self):
        """获取当前机械臂位姿"""
        DianaApi.getTcpPos(self.poses, self.ip_address)
        angle_euler = self.poses[3:]
        # 转化为欧拉角
        DianaApi.axis2RPY(angle_euler)
        # 转化为毫米
        for i in range(3):
            self.poses[i] *= 1000.0
        for i in range(3):
            # 将旋转角度从弧度转换为角度
            angle_euler[i] = math.degrees(angle_euler[i])
        self.poses[3:] = angle_euler
        return self.poses
        
    def move_to_pose(self, poses):
        """移动机械臂到指定位姿"""
        result = DianaApi.moveJToPose(poses, self.velocity, self.acceleration, self.ip_address)
        if not result:
            e = DianaApi.getLastError()
            e_info = DianaApi.formatError(e)
            print(f'机械臂移动失败,错误码为：{e},错误描述信息为：{e_info}')
        return result
        
    def set_gripper_position(self, position):
        """设置夹爪位置"""
        if self.gripper is None:
            print("夹爪未初始化")
            return False
            
        position = max(0, min(1000, position))  # 确保在有效范围内
        
        try:
            self.gripper.Position(position)
            self.current_gripper_position = position
            return True
        except Exception as e:
            print(f"夹爪控制失败: {str(e)}")
            return False
            
    def reset_to_initial_position(self):
        """重置机械臂到初始位置"""
        try:
            PickAndPlace.arm_pose_init(self.velocity, self.acceleration, self.ip_address)
            PickAndPlace.wait_move()
            return True
        except Exception as e:
            print(f"重置机械臂位置失败: {str(e)}")
            return False
            
    def stop_robot(self):
        """停止机器人"""
        try:
            DianaApi.stop(self.ip_address)
            DianaApi.destroySrv(self.ip_address)
            return True
        except Exception as e:
            print(f"停止机械臂失败: {str(e)}")
            return False
            
    def lift_object(self, lift_height=50):
        """抬起物体"""
        current_pose = self.get_current_pose()
        # 抬高指定高度
        lift_pose = (
            current_pose[0], current_pose[1], current_pose[2] + lift_height,
            current_pose[3], current_pose[4], current_pose[5]
        )
        return self.move_to_pose(lift_pose)
        
    def get_place_position(self, location_keyword):
        """根据位置关键词获取放置坐标"""
        place_positions = {
            "左边": (200, -150, 200, 0.0, 0.0, 0.0),
            "右边": (200, 150, 200, 0.0, 0.0, 0.0),
            "前面": (300, 0, 200, 0.0, 0.0, 0.0),
            "后面": (100, 0, 200, 0.0, 0.0, 0.0),
            "中间": (200, 0, 200, 0.0, 0.0, 0.0),
            "桌子": (250, 100, 200, 0.0, 0.0, 0.0),
            "工作台": (250, 100, 200, 0.0, 0.0, 0.0),
            "盒子": (150, -100, 150, 0.0, 0.0, 0.0),
            "箱子": (150, -100, 150, 0.0, 0.0, 0.0)
        }
        
        for keyword, position in place_positions.items():
            if keyword in location_keyword:
                return position
                
        # 默认位置
        return (300, 150, 200, 0.0, 0.0, 0.0)
        
    def execute_pick_and_place(self, target_location=None):
        """执行完整的抓取和放置动作"""
        try:
            # 确保夹爪抓取
            if self.current_gripper_position > 500:
                self.set_gripper_position(300)
                time.sleep(1.5)
                print("夹爪已关闭，物体已抓取")
            
            # 抬起物体
            print("抬起物体...")
            self.lift_object(50)
            time.sleep(2)
            
            # 确定放置位置
            if target_location:
                place_pose = self.get_place_position(target_location)
                print(f"根据语音指令放置到: {target_location}")
            else:
                place_pose = (300, 150, 200, 0.0, 0.0, 0.0)
                print("使用默认放置位置")
            
            # 移动到放置位置
            print(f"移动到放置位置: {place_pose}...")
            self.move_to_pose(place_pose)
            time.sleep(2)
            
            # 放下物体
            print("放下物体...")
            self.set_gripper_position(1000)
            time.sleep(1.5)
            print("物体已放置")
            
            return True
            
        except Exception as e:
            print(f"抓取放置执行失败: {str(e)}")
            return False 