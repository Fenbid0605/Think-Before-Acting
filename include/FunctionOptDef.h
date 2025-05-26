#pragma once

extern "C" typedef enum _FunctionIndex {
    T_FREEDRIVING = 0,
    T_CART_IMPEDANCE = 1,
} function_index_e;

extern "C" typedef enum _FreeDrivingDirection {
    T_ALLOW_X = 0x01,
    T_ALLOW_Y = 0x02,
    T_ALLOW_Z = 0x04,
    T_ALLOW_XYZ = 0x07,
    T_ALLOW_RX = 0x10,
    T_ALLOW_RY = 0x20,
    T_ALLOW_RZ = 0x40,
    T_ALLOW_RXYZ = 0x70,
} free_driving_direction_e;

extern "C" typedef enum _FunctionBaseName {
    T_DEFAULT_OPT = 0,                      // set only
    T_FREEDRIVING_OPT_OFFSET = 0x10000,     // not used
    T_CART_IMPEDANCE_OPT_OFFSET = 0x10100,  // not used
} function_opt_name_e;

extern "C" typedef enum _FreeDrivingOptName {
    T_ALLOW_DIRECTION = function_opt_name_e::T_FREEDRIVING_OPT_OFFSET,  // read and write
} freedriving_opt_name_e;

extern "C" typedef enum _CartImpedanceOptName {
    T_COORDINATE_TYPE = function_opt_name_e::T_CART_IMPEDANCE_OPT_OFFSET,  // read and write
    T_IS_SINGLE_AXIS_LOCKED,                                               // read only
    T_LOCKED_SINGLE_AXIS,                                                  // write only
} cart_impedance_opt_name_e;
