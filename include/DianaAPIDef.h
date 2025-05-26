#pragma once

#pragma pack(1)

#include <stdint.h>
#ifndef _WIN32
#include <sys/types.h>
#endif

#define _API_SUPPORT_V2_

#define MAX_SUPPORT_ROBOTARM_NUM 3

extern "C" typedef void (*FNCERRORCALLBACK)(int e, const char *strIpAddress);

// TrajectoryState
extern "C" typedef struct _TrajectoryState {
    int taskId;
    int segCount;
    int segIndex;
    int errorCode;
    int isControllerPaused;
    int isSafetyDriving;
    int isFreeDriving;
    int isZeroSpaceFreeDriving;
    int isRobotHoldBrake;
    int isProgramRunningOrPause;
    int isTeachPendantPaused;
    int isControllerTerminated;
} StrTrajectoryState;

extern "C" typedef struct _ErrorInfo {
    int errorId;
    int errorType;
    int errorCode;
    char errorMsg[64];
} StrErrorInfo;
//用户层机械臂反馈状态结构体
extern "C" typedef struct _RobotStateInfo {
    double jointPos[7];
    double jointAngularVel[7];
    double jointCurrent[7];
    double jointTorque[7];
    double tcpPos[6];
    double tcpExternalForce;
    bool bCollision;
    bool bTcpForceValid;
    double tcpForce[6];
    double jointForce[7];
    StrTrajectoryState trajState;
    StrErrorInfo errorInfo;
} StrRobotStateInfo;

#define USER_MAXIMUM_DOUBLE_SIZE 40
#define USER_MAXIMUM_INT8_SIZE 160

extern "C" typedef struct _CustomStateInfo {
    double dblField[USER_MAXIMUM_DOUBLE_SIZE];
    int8_t int8Field[USER_MAXIMUM_INT8_SIZE];
} StrCustomStateInfo;

extern "C" typedef void (*FNCSTATECALLBACK)(StrRobotStateInfo *pinfo, const char *strIpAddress);

#define    ROBOTSTATE_CUSTOM_BASIC                (0x00000001)    //基本状态信息，必须推送，不可定制，包括 BasicRobotState_1 中 stateBits, trajectoryState及 BasicRobotState2 中所有数据
#define    ROBOTSTATE_CUSTOM_JOINTPOS             (0x00000002)    //关节反馈位置
#define    ROBOTSTATE_CUSTOM_LINKJOINTPOS         (0x00000004)    //低速侧关节反馈位置
#define    ROBOTSTATE_CUSTOM_JOINTANGULARVEL      (0x00000008)     // 关节反馈速度
#define    ROBOTSTATE_CUSTOM_JOINTCURRENT         (0x00000010)    // 关节反馈电流
#define    ROBOTSTATE_CUSTOM_ORIGINJOINTTORQUE    (0x00000020)    // 关节反馈力矩(含零偏)
#define    ROBOTSTATE_CUSTOM_JOINTTORQUEOFFSET    (0x00000040)    //关节扭矩传感器零偏
#define    ROBOTSTATE_CUSTOM_JOINTFORCE           (0x00000080)    // 关节外力
#define    ROBOTSTATE_CUSTOM_TCPFORCE             (0x00000100)    // 末端受到的外力

#define    ROBOTSTATE_CUSTOM_ALL                  (0xffffffff)    //定制所有

enum _CustomRobotStateAction {
    API_CUSTOM_ADD = 0,    //定制
    API_CUSTOM_DEL = 1,    //取消
    API_CUSTOM_RESET = 2,    //重置
};

#pragma pack()