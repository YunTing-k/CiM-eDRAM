"""
本文件中的函数与模型仿真不相关
包含函数有按照一定颜色和格式输出相关语句，
模型转换MAT文件
权值数据可视化
"""
import sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io as sio


def print_console(str_in, m1, fg, bg, m2):
    """给定当前和后继显示方式 前景色 背景色 打印字符串到console"""
    # print('\033[' + str(m1) + ';' + str(fg) + ';' + str(bg) + 'm' + str_in + '\033[' + str(m2) + 'm')
    tqdm.write('\033[' + str(m1) + ';' + str(fg) + ';' + str(bg) + 'm' + str_in + '\033[' + str(m2) + 'm',
               file=sys.stderr)


def print_error(str_in, file_name):
    """打印错误信息"""
    # print('\033[1;31;40m[Error]:' + str_in + '\033[0;31;40m >>in file:\033[4;31;40m' + file_name + '\033[0m')
    tqdm.write('\033[1;31;40m[Error]:' + str_in + '\033[0;31;40m >>in file:\033[4;31;40m' + file_name + '\033[0m',
               file=sys.stderr)


def print_warn(str_in, file_name):
    """打印警告信息"""
    # print('\033[1;33;40m[Warn]:' + str_in + '\033[0;33;40m >>in file:\033[4;33;40m' + file_name + '\033[0m')
    tqdm.write('\033[1;33;40m[Warn]:' + str_in + '\033[0;33;40m >>in file:\033[4;33;40m' + file_name + '\033[0m',
               file=sys.stderr)


def print_info(str_in, file_name):
    """打印提示信息"""
    # print('\033[1;35;40m[Info]:' + str_in + '\033[0;35;40m >>in file:\033[4;35;40m' + file_name + '\033[0m')
    tqdm.write('\033[1;35;40m[Info]:' + str_in + '\033[0;35;40m >>in file:\033[4;35;40m' + file_name + '\033[0m',
               file=sys.stderr)


def print_debug(str_in, file_name):
    """打印提示信息"""
    # print('\033[1;32;40m[Debug]:' + str_in + '\033[0;32;40m >>in file:\033[4;32;40m' + file_name + '\033[0m')
    tqdm.write('\033[1;32;40m[Debug]:' + str_in + '\033[0;32;40m >>in file:\033[4;32;40m' + file_name + '\033[0m',
               file=sys.stderr)


def print_param(str_in):
    """打印参数信息"""
    # print('\033[1;34;40m' + str_in + '\033[0m')
    tqdm.write('\033[1;34;40m' + str_in + '\033[0m',
               file=sys.stderr)


def mat_to_numpy(file, name):
    data = sio.loadmat(file)
    curve_cluster = data[name]  # 数据
    return curve_cluster


def weight_to_mat(net_dic, name):
    """保存网络参数到MAT文件"""
    data = {}  # 创建空字典
    for dict_key in net_dic.keys():
        layer_name = dict_key.split('.')
        layer_name = '{}_{}'.format(layer_name[0], layer_name[1][0])
        mat = net_dic[dict_key].cpu().numpy()
        data[layer_name] = mat
        sio.savemat('./MATfile/' + name + '.mat', data)


def numpy_to_mat(data_in, name):
    """保存numpy数组到MAT文件"""
    data = {'data': data_in}  # 创建空字典
    sio.savemat('./MATfile/' + name + '.mat', data)


def tensor_to_mat(data_in, name):
    """保存tensor到MAT文件"""
    data = {'data': data_in.cpu().numpy()}  # 创建空字典
    sio.savemat('./MATfile/' + name + '.mat', data)


def display_distribution(param, display_mode, global_config, bin_num, dpi, if_save, name, if_active):
    """显示数据分布 根据显示模式进行可视化
    param:输入网络权值参数
    display_mode 0:不显示 1:显示所有数据分布 2:显示非0数据分布
    if_save True:直接保存不展示 False:展示不保存
    global_config all:展示全局的分布 single:展示单个layer的分布
    if_active:是否启用 true启用，false不启用"""
    if not if_active:
        return
    elif display_mode == 0:
        return
    elif display_mode == 1:
        if global_config == 'all':
            file_name = name + '-All overall weight'
            params_list = np.empty((0, 1), float)
            for _, layer_param in enumerate(param):
                tmp_data = layer_param.data.cpu().numpy().reshape(-1, 1)
                params_list = np.vstack((params_list, tmp_data))
            plt.hist(params_list[:], bins=bin_num)
            if if_save:
                plt.savefig('./Figure/' + file_name + '.png', dpi=dpi, pad_inches=0.0)
                plt.close()
            else:
                plt.show()
        elif global_config == 'single':
            file_name = name + '-Single overall weight'
            for layer_num, layer_param in enumerate(param):
                tmp_data = layer_param.data.cpu().numpy().reshape(-1, 1)
                plt.hist(tmp_data, bins=bin_num)
                if if_save:
                    plt.savefig('./Figure/' + file_name + 'in layer-' + str(layer_num+1) + '.png', dpi=dpi, pad_inches=0)
                    plt.close()
                else:
                    plt.show()
    elif display_mode == 2:
        if global_config == 'all':
            file_name = name + '-All nonzero weight'
            params_list = np.empty((0, 1), float)
            for _, layer_param in enumerate(param):
                tmp_data = layer_param.data.cpu().numpy().reshape(-1, 1)
                # tmp_data = tmp_data[tmp_data != 0].reshape(-1, 1)  # 去除所有的0
                tmp_data = tmp_data[np.abs(tmp_data) >= 1e-3].reshape(-1, 1)  # 去除所有的0
                params_list = np.vstack((params_list, tmp_data))
            plt.hist(params_list[:], bins=bin_num)
            if if_save:
                plt.savefig('./Figure/' + file_name + '.png', dpi=dpi, pad_inches=0.0)
                plt.close()
            else:
                plt.show()
        elif global_config == 'single':
            file_name = name + '-Single nonzero weight'
            for layer_num, layer_param in enumerate(param):
                tmp_data = layer_param.data.cpu().numpy().reshape(-1, 1)
                # tmp_data = tmp_data[tmp_data != 0]  # 去除所有的0
                tmp_data = tmp_data[np.abs(tmp_data) >= 1e-3]  # 去除所有的0
                plt.hist(tmp_data, bins=bin_num)
                if if_save:
                    plt.savefig('./Figure/' + file_name + 'in layer-' + str(layer_num+1) + '.png', dpi=dpi, pad_inches=0)
                    plt.close()
                else:
                    plt.show()
