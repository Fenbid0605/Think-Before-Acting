from fastmcp import FastMCP, Context
import sys
import os
import time

# 添加 bin 目录到 Python 路径以导入 DianaApi
current_dir = os.path.dirname(os.path.abspath(__file__))
bin_dir = os.path.join(current_dir, "bin")
sys.path.append(bin_dir)

from DianaApi import (
    initSrv, formatError, getJointPos, getTcpPos, getRobotState,
    moveJ, moveL, moveTCP, rotationTCP, stop, resume, freeDriving,
    freedriving_mode_e, FNCERRORCALLBACK, getLastError, releaseBrake, holdBrake
)

def errorCallback(e, ip):
    """
    错误回调函数，用于处理API调用中的错误
    
    当机器人API调用发生错误时，该函数会被调用，打印错误代码
    
    参数:
        e (int): 错误代码，参考DianaApi.py中的errorCodeMessage字典
        ip (str): 机器人IP地址，用于标识发生错误的机器人控制器
    """
    print("error code:" + str(e))


# def robotStateCallback(stateInfo, ip):
#     """
#     机器人状态回调函数，用于接收和显示机器人的当前状态信息
    
#     周期性地接收机器人的状态数据，包括关节角度、角速度、电流和扭矩等
    
#     参数:
#         stateInfo (StrRobotStateInfo): 包含机器人状态的结构体，由底层API填充
#         ip (str): 机器人IP地址，用于标识机器人控制器
#     """
#     for i in range(0, 7):
#         print('关节角度数组:{0}'.format(stateInfo.contents.jointPos[i]))
#     for i in range(0, 7):
#         print('关节角速度数组{0}'.format(stateInfo.contents.jointAngularVel[i]))
#     for i in range(0, 7):
#         print('关节角电流当前值数组{0}'.format(stateInfo.contents.jointCurrent[i]))
#     for i in range(0, 7):
#         print('关节角扭矩数组{0}'.format(stateInfo.contents.jointTorque[i]))

# 创建 MCP 服务器
mcp = FastMCP("Diana Robot MCP Server")


@mcp.tool()
async def get_joint_positions(ctx: Context):
    """获取机器人当前关节位置
    
    Returns:
        七个关节的角度值（弧度）
    """
    joint_pos = [0.0] * 7
    result = getJointPos(joint_pos)
    return {"success": True, "joint_positions": joint_pos}

@mcp.tool()
async def get_tcp_position(ctx: Context):
    """获取机器人当前 TCP 位置
    
    Returns:
        TCP 的位置和姿态，格式为 [x, y, z, rx, ry, rz]
    """
    tcp_pos = [0.0] * 6
    result = getTcpPos(tcp_pos)
    return {"success": True, "tcp_position": tcp_pos}

@mcp.tool()
async def get_robot_state(ctx: Context):
    """获取机器人当前状态
    
    Returns:
        机器人状态值
    """
    state = getRobotState()
    return {"success": True, "state": state}

@mcp.tool()
async def move_joints(positions: list, velocity: float, acceleration: float, ctx: Context):
    """移动机器人到指定的关节位置
    
    Args:
        positions: 七个关节的目标位置（弧度）
        velocity: 速度比例（0.0-1.0）
        acceleration: 加速度比例（0.0-1.0）
    
    Returns:
        移动操作结果
    """
    await ctx.info(f"移动关节到位置: {positions}, 速度: {velocity}, 加速度: {acceleration}")
    
    if len(positions) != 7:
        return {"success": False, "message": "必须提供七个关节位置值"}
    
    result = moveJ(positions, velocity, acceleration)
    return {"success": True, "message": "移动指令已发送", "result": result}

@mcp.tool()
async def move_linear(position: list, velocity: float, acceleration: float, ctx: Context):
    """线性移动机器人到指定的 TCP 位置
    
    Args:
        position: TCP 的目标位置和姿态 [x, y, z, rx, ry, rz]
        velocity: 速度值（mm/s）
        acceleration: 加速度值（mm/s²）
    
    Returns:
        移动操作结果
    """
    await ctx.info(f"线性移动到位置: {position}, 速度: {velocity}, 加速度: {acceleration}")
    
    if len(position) != 6:
        return {"success": False, "message": "必须提供六个 TCP 位置值"}
    
    result = moveL(position, velocity, acceleration)
    return {"success": True, "message": "移动指令已发送", "result": result}

@mcp.tool()
async def move_tcp_direction(direction: int, velocity: float, acceleration: float, ctx: Context):
    """向指定方向移动 TCP
    
    Args:
        direction: 移动方向（0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-）
        velocity: 速度值（mm/s）
        acceleration: 加速度值（mm/s²）
    
    Returns:
        移动操作结果
    """
    directions = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"]
    await ctx.info(f"沿 {directions[direction]} 方向移动 TCP, 速度: {velocity}, 加速度: {acceleration}")
    
    from DianaApi import tcp_direction_e
    direction_enum = tcp_direction_e(direction)
    result = moveTCP(direction_enum, velocity, acceleration)
    return {"success": True, "message": "移动指令已发送", "result": result}

@mcp.tool()
async def rotate_tcp_direction(direction: int, velocity: float, acceleration: float, ctx: Context):
    """围绕指定轴旋转 TCP
    
    Args:
        direction: 旋转方向（0=Rx+, 1=Rx-, 2=Ry+, 3=Ry-, 4=Rz+, 5=Rz-）
        velocity: 角速度（rad/s）
        acceleration: 角加速度（rad/s²）
    
    Returns:
        旋转操作结果
    """
    directions = ["Rx+", "Rx-", "Ry+", "Ry-", "Rz+", "Rz-"]
    await ctx.info(f"围绕 {directions[direction]} 轴旋转 TCP, 速度: {velocity}, 加速度: {acceleration}")
    
    from DianaApi import tcp_direction_e
    direction_enum = tcp_direction_e(direction)
    result = rotationTCP(direction_enum, velocity, acceleration)
    return {"success": True, "message": "旋转指令已发送", "result": result}

@mcp.tool()
async def stop_robot(ctx: Context):
    """停止机器人所有运动
    
    Returns:
        停止操作结果
    """
    await ctx.info("停止机器人所有运动...")
    result = stop()
    return {"success": True, "message": "停止指令已发送", "result": result}

@mcp.tool()
async def resume_robot(ctx: Context):
    """恢复机器人运动
    
    Returns:
        恢复操作结果
    """
    await ctx.info("恢复机器人运动...")
    result = resume()
    return {"success": True, "message": "恢复指令已发送", "result": result}

@mcp.tool()
async def enable_freedriving(mode: int, ctx: Context):
    """启用自由驱动模式
    
    Args:
        mode: 自由驱动模式 (0=禁用, 1=普通, 2=强制)
    
    Returns:
        操作结果
    """
    modes = ["禁用", "普通", "强制"]
    await ctx.info(f"设置自由驱动模式: {modes[mode]}")
    
    mode_enum = freedriving_mode_e(mode)
    result = freeDriving(mode_enum)
    return {"success": True, "message": f"已设置为{modes[mode]}自由驱动模式", "result": result}

@mcp.tool()
async def release_brake(ctx: Context):
    """释放抱闸
    
    Returns:
        释放抱闸操作结果
    """
    result = releaseBrake()
    return {"success": True, "message": "抱闸已释放", "result": result}

@mcp.tool()
async def hold_brake(ctx: Context):
    """抱闸
    
    Returns:  
        抱闸操作结果
    """
    result = holdBrake()
    return {"success": True, "message": "抱闸已抱紧", "result": result}

@mcp.tool()
async def connect_robot(robot_ip: str, ctx: Context):
    """连接到 Diana 机器人
    
    Args:
        robot_ip: 机器人的 IP 地址
    
    Returns:
        连接结果
    """
    await ctx.info(f"正在连接到机器人 {robot_ip}...")
    
    from DianaApi import SRV_NET_ST
    from ctypes import byref, c_char
    
    # 创建并初始化网络结构体
    # 根据SRV_NET_ST结构体定义，第一个字段是字节数组
    srv_net_st_obj = SRV_NET_ST()
    srv_net_st_obj.ipAddress = robot_ip.encode('utf-8')
    srv_net_st_obj.SLocHeartbeatPortrvIp = 0
    srv_net_st_obj.LocRobotStatePort = 0
    srv_net_st_obj.LocSrvPort = 0
    srv_net_st_obj.LocRealtimeSrvPort = 0
    srv_net_st_obj.LocPassThroughSrvPort = 0
    
    # 初始化连接
    result = initSrv(byref(srv_net_st_obj), None, None)
    
    if result:
        await ctx.info('连接成功!')
        return {"success": True, "message": "已成功连接到机器人"}
    else:
        time.sleep(0.1)
        e = getLastError()  # 获取最近的错误代码
        e_info = formatError(e)  # 获取错误的描述信息
        error_message = f'连接失败,错误码为：{e},错误描述信息为：{e_info}'
        await ctx.info(error_message)
        return {"success": False, "message": error_message}

# 运行服务器
if __name__ == "__main__":
    print("正在启动 Diana 机器人 MCP 服务器 (stdio 模式)...")
    # 连接到机器人控制器
    netInfo = ('192.168.10.75', 0, 0, 0, 0, 0)  # 网络连接信息，包含IP地址和其他参数
    fnError = FNCERRORCALLBACK(errorCallback)  # 注册错误回调函数
    # fnState = FNCSTATECALLBACK(robotStateCallback)  # 注册状态回调函数
    ERROR_OK = initSrv(netInfo, fnError)  # 初始化服务连接
    if ERROR_OK == True:
        print('连接成功!')
    else:
        time.sleep(0.1)
        e = getLastError()  # 获取最近的错误代码
        e_info = formatError(e)  # 获取错误的描述信息
        print('连接失败,错误码为：{0},错误描述信息为：{1}'.format(e, e_info))
        sys.exit()

    # 运行服务器，使用 stdio 模式
    mcp.run(transport="stdio")
