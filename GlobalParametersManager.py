"""
这是一个跨文件的全局参数管理器
功能为参数全局传递，修改，删除，保存和加载
"""
import os
import numpy as np
import Utilities as ut

current_name = os.path.basename(__file__)  # 当前模块名字


def init():
    """全局参数管理器初始化"""
    global _global_dict
    _global_dict = {}
    ut.print_info('Global parameters dictionary created', current_name)


def set_param(key, value):
    """定义一个全局参数"""
    _global_dict[key] = value


def get_param(key):
    """获得一个全局参数，不存在则提示读取对应参数失败"""
    try:
        return _global_dict[key]
    except KeyError:
        _str = 'Read parameter”' + key + '“failed!'
        ut.print_error(_str, current_name)
        return 0


def print_param(key):
    """输出指定参数"""
    try:
        param_type = str(type(_global_dict[key])).replace('<class ', '').replace('>', '')
        _str = '"' + key + '"=' + str(_global_dict[key]) + '  Type=' + param_type
        ut.print_param(_str)
    except KeyError:
        _str = 'Print parameter”' + key + '“failed!'
        ut.print_error(_str, current_name)


def print_all_param():
    """输出所有参数"""
    ut.print_info('Start printing all parameters', current_name)
    for key in _global_dict:
        param_type = str(type(_global_dict[key])).replace('<class ', '').replace('>', '')
        _str = '"' + key + '"=' + str(_global_dict[key]) + '  Type=' + param_type
        ut.print_param(_str)
    ut.print_info('All parameters printed', current_name)


def write_param():
    """输出参数到文件"""
    file_name = './Parameters/' + get_param('Train_dataset_type') + 'AllParam.npy'
    np.save(file_name, _global_dict)
    ut.print_info('Params saved in' + file_name, current_name)


def load_param():
    """从文件中读取参数"""
    try:
        global _global_dict
        _global_dict = np.load('./Parameters/AllParamsIn.npy', allow_pickle=True)
        ut.print_info('Params loaded from /Parameters/AllParamsIn.npy', current_name)
        print_all_param()
    except FileNotFoundError:
        ut.print_error('Nonexistent input parameters file!', current_name)


def delete_param(key):
    """删除指定参数"""
    try:
        _global_dict.pop(key)
    except KeyError:
        _str = 'Delete parameter”' + key + '“failed!'
        ut.print_error(_str, current_name)


def clear_param():
    """删除所有参数"""
    _global_dict.clear()
    ut.print_info('All parameters deleted', current_name)
