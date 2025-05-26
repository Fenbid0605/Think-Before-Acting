#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块测试脚本 - 验证各模块是否可以正常导入和基本功能
"""

def test_imports():
    """测试所有模块的导入"""
    print("开始测试模块导入...")
    
    try:
        import config
        print("✓ config.py 导入成功")
    except ImportError as e:
        print(f"✗ config.py 导入失败: {e}")
        return False
    
    try:
        from audio_processing import AudioBuffer, RealtimeTranscriber, SpeakTTS
        print("✓ audio_processing.py 导入成功")
    except ImportError as e:
        print(f"✗ audio_processing.py 导入失败: {e}")
        return False
    
    try:
        from vision_processing import CameraManager, ObjectDetector
        print("✓ vision_processing.py 导入成功")
    except ImportError as e:
        print(f"✗ vision_processing.py 导入失败: {e}")
        return False
    
    try:
        from robot_control import RobotController
        print("✓ robot_control.py 导入成功")
    except ImportError as e:
        print(f"✗ robot_control.py 导入失败: {e}")
        return False
    
    try:
        from tactile_sensor import TactileSensor, TactileFeedbackController
        print("✓ tactile_sensor.py 导入成功")
    except ImportError as e:
        print(f"✗ tactile_sensor.py 导入失败: {e}")
        return False
    
    try:
        from mcp_tools import MCPToolsManager
        print("✓ mcp_tools.py 导入成功")
    except ImportError as e:
        print(f"✗ mcp_tools.py 导入失败: {e}")
        return False
    
    print("所有模块导入测试通过！")
    return True

def test_config():
    """测试配置模块"""
    print("\n测试配置模块...")
    
    try:
        import config
        
        # 检查关键配置是否存在
        assert hasattr(config, 'DEFAULT_IP_ADDRESS'), "缺少 DEFAULT_IP_ADDRESS"
        assert hasattr(config, 'CAMERA_MATRIX'), "缺少 CAMERA_MATRIX"
        assert hasattr(config, 'TRIGGER_KEYWORDS'), "缺少 TRIGGER_KEYWORDS"
        assert hasattr(config, 'OBJECT_TYPES'), "缺少 OBJECT_TYPES"
        
        print(f"✓ 默认IP地址: {config.DEFAULT_IP_ADDRESS}")
        print(f"✓ 触发关键词数量: {len(config.TRIGGER_KEYWORDS)}")
        print(f"✓ 物体类型数量: {len(config.OBJECT_TYPES)}")
        print("✓ 配置模块测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置模块测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        # 测试音频缓冲区
        from audio_processing import AudioBuffer
        buffer = AudioBuffer(min_length=1000)
        print("✓ AudioBuffer 创建成功")
        
        # 测试触觉传感器
        from tactile_sensor import TactileSensor
        sensor = TactileSensor()
        print("✓ TactileSensor 创建成功")
        
        # 测试机器人控制器
        from robot_control import RobotController
        controller = RobotController()
        print("✓ RobotController 创建成功")
        
        # 测试MCP工具管理器
        from mcp_tools import MCPToolsManager
        manager = MCPToolsManager()
        print("✓ MCPToolsManager 创建成功")
        
        print("✓ 基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Diana机器人MCP服务器 - 模块化测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_imports,
        test_config,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # 输出测试结果
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！模块化结构正常工作。")
        print("\n可以运行以下命令启动服务器:")
        print("python main_server.py")
    else:
        print("❌ 部分测试失败，请检查模块依赖和配置。")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 