"""定义了器件相关的函数和参数"""
import copy
import os
import Utilities as ut
import GlobalParametersManager as gpm
import numpy as np
import torch
import scipy.io as io

current_name = os.path.basename(__file__)  # 当前模块名字


def set_default_param():
    """设置默认参数"""
    gpm.set_param('Include_Noise', False)  # 是否考虑噪声
    gpm.set_param('Mapping_Strategy', 'Unroll')  # 映射方式 影响最后的噪声评估结果 Unroll:权重对应多个器件
    # gpm.set_param('Mapping_Strategy', 'Fold')  # 映射方式 影响最后的噪声评估结果 Fold:认为权重和器件一一对应
    gpm.set_param('Noise_Method', 'PerStatus')  # 噪声类别 PerStatus: 状态噪声分布不同 Overall: 状态噪声分布一致
    gpm.set_param('Discrete_Calibration_order', 0)  # 离散器件的标定级数 CIS / CIM 离散器件级数为0时 默认为均值"标定"
    """CIM器件配置"""
    gpm.set_param('Device_File_ideal', './Device/ZnO/4bit/ideal/')  # 曲线器件响应数据的路径 单位 S 1/Ω
    gpm.set_param('Device_File_nonideal', './Device/ZnO/4bit/nonideal/')  # 曲线器件响应数据的路径
    gpm.set_param('Device_File_area', './Device/ZnO/4bit/')  # 器件面积数据
    gpm.set_param('Performance_estimate', True)  # 是否估计性能参数
    gpm.set_param('Max_vin', 0.1)  # 最大输入电压 V
    gpm.set_param('Store_voltage', 3)  # 权重存储电压
    gpm.set_param('If_Area_Scale', True)
    gpm.set_param('Width_Scale_factor', 0.5 * 1e-3)  # 缩放比例 FET 宽度 缩放比例
    gpm.set_param('Length_Scale_Factor', 1e-3)  # 缩放比例 FET 沟道长度 缩放比例
    gpm.set_param('If_Power_Scale', True)
    gpm.set_param('Conductance_Scale_Factor',  # 电导缩放比例
                  gpm.get_param('Width_Scale_factor') / gpm.get_param('Length_Scale_Factor'))
    gpm.set_param('Clock_freq', 100000000)  # 运算频率 Hz
    gpm.set_param('Unit_cap', 8.02e-3)  # 单位电容 pF/um2
    gpm.set_param('Read_Area', 3200)  # um2 40*10*4*2
    gpm.set_param('Refresh_rate', [1/10000, 100000])  # 权重刷新频率 Hz [不进行频繁刷新的场景 进行频繁刷新的场景]
    ut.print_info('Device parameters added', current_name)


def discrete_calibrate_curve(data_path, calibration_order):
    """读取原始数据，根据输入阶数标定标定曲线，返回标定后的离散值，总状态数目(CIM)
    适用于器件本身离散情况"""
    # 读取数据
    data = io.loadmat(data_path)
    curve_cluster = data['curve_cluster']  # 数据
    """器件状态数"""
    device_status_num = curve_cluster.shape[0]
    curve_num = curve_cluster.shape[1]  # 曲线数目
    if calibration_order != 0:  # 不进行均值标定时
        # 进行标定
        x = np.zeros(device_status_num * curve_num)
        y = np.zeros(device_status_num * curve_num)
        for i in range(curve_num):
            x[i * device_status_num:(i + 1) * device_status_num] = np.arange(0, device_status_num)
            y[i * device_status_num:(i + 1) * device_status_num] = curve_cluster[:, i]
        poly = np.polyfit(x, y, deg=calibration_order)
        """标定散点"""
        calibration_curve = np.polyval(poly, np.arange(0, device_status_num))
        SSE = 0
        SST = 0
        curve_mean = curve_cluster.mean()  # 数据均值
        for i in range(curve_num):
            SSE = SSE + np.sum((calibration_curve - curve_cluster[:, i]) ** 2)
            SST = SST + np.sum((curve_cluster[:, i] - curve_mean) ** 2)
        r_square = 1 - SSE / SST  # 确定系数
        ut.print_info('Device curve calibration done', current_name)

        abs_space_distance = 0  # 与线性情况的绝对空间距离
        a = (calibration_curve[device_status_num - 1] - calibration_curve[0]) / (device_status_num - 1)
        b = calibration_curve[0]
        for i in range(device_status_num):
            abs_space_distance += abs(calibration_curve[i] - (a * i + b)) / (a * i + b)
        abs_space_distance /= device_status_num
        calibration_curve_diff = np.diff(calibration_curve)
        abs_diff_distance = np.sum(np.abs(calibration_curve_diff - a)) / (device_status_num - 1)
        ut.print_param('Calibration parameters,R-square:%.4f,Absolute space distance:%.4e,Absolute diff distance:%.4e' %
                       (r_square, abs_space_distance, abs_diff_distance))
        return curve_cluster, device_status_num, calibration_curve, r_square, abs_space_distance, abs_diff_distance
    else:  # 进行均值标定
        # 进行标定
        calibration_curve = np.zeros(device_status_num)
        for i in range(device_status_num):
            calibration_curve[i] = np.mean(curve_cluster[i, :])
        SSE = 0
        SST = 0
        curve_mean = curve_cluster.mean()  # 数据均值
        for i in range(curve_num):
            SSE = SSE + np.sum((calibration_curve - curve_cluster[:, i]) ** 2)
            SST = SST + np.sum((curve_cluster[:, i] - curve_mean) ** 2)
        r_square = 1 - SSE / SST  # 确定系数
        ut.print_info('Device curve calibration with 0 order done', current_name)

        abs_space_distance = 0  # 与线性情况的绝对空间距离
        a = (calibration_curve[device_status_num - 1] - calibration_curve[0]) / (device_status_num - 1)
        b = calibration_curve[0]
        for i in range(device_status_num):
            abs_space_distance += abs(calibration_curve[i] - (a * i + b)) / (a * i + b)
        abs_space_distance /= device_status_num
        calibration_curve_diff = np.diff(calibration_curve)
        abs_diff_distance = np.sum(np.abs(calibration_curve_diff - a)) / (device_status_num - 1)
        ut.print_param('Calibration parameters,R-square:%.4f,Absolute space distance:%.4e,Absolute diff distance:%.4e' %
                       (r_square, abs_space_distance, abs_diff_distance))
        return curve_cluster, device_status_num, calibration_curve, r_square, abs_space_distance, abs_diff_distance


def continuous_calibrate_curve(data_path, bit_num, calibration_order):
    """根据bit数和指定标定级数来得到连续器件的标定值
    适用于连续情况"""
    # 读取数据
    data = io.loadmat(data_path)
    curve_cluster = data['curve_cluster']  # 数据 INPUT-WEIGHT
    control_list = data['control_list']  # 调控数据 如对应的电压值
    """器件状态数"""
    control_status_num = curve_cluster.shape[0]  # 输入的调控数目 如电压数
    curve_num = curve_cluster.shape[1]  # 曲线数目
    # 进行标定
    x = np.zeros(control_status_num * curve_num)
    y = np.zeros(control_status_num * curve_num)
    for i in range(curve_num):
        x[i * control_status_num:(i + 1) * control_status_num] = np.squeeze(control_list)  # 以control数据作为X
        y[i * control_status_num:(i + 1) * control_status_num] = curve_cluster[:, i]
    poly = np.polyfit(x, y, deg=calibration_order)  # 标定
    """与原始控制数目相同的的标定值 用于噪声拟合"""
    all_control = np.squeeze(control_list)
    all_calibration_curve = np.polyval(poly, all_control)
    """标定散点"""
    cali_control = np.linspace(control_list[0], control_list[-1], 2 ** bit_num)  # 以间隔的control数据作为X 间隔由bit_num指定
    calibration_curve = np.polyval(poly, cali_control)
    _calibration_curve = np.polyval(poly, np.squeeze(control_list))
    SSE = 0
    SST = 0
    curve_mean = curve_cluster.mean()  # 数据均值
    for i in range(curve_num):
        SSE = SSE + np.sum((_calibration_curve - curve_cluster[:, i]) ** 2)
        SST = SST + np.sum((curve_cluster[:, i] - curve_mean) ** 2)
    r_square = 1 - SSE / SST  # 确定系数
    ut.print_info('Device curve calibration done', current_name)

    _control_status_num = 2 ** bit_num
    abs_space_distance = 0  # 与线性情况的绝对空间距离
    a = (calibration_curve[_control_status_num - 1] - calibration_curve[0]) / (_control_status_num - 1)
    b = calibration_curve[0]
    for i in range(_control_status_num):
        abs_space_distance += abs(calibration_curve[i] - (a * i + b)) / (a * i + b)
    abs_space_distance /= _control_status_num
    calibration_curve_diff = np.diff(np.squeeze(calibration_curve))
    abs_diff_distance = np.sum(np.abs(calibration_curve_diff - a)) / (_control_status_num - 1)
    ut.print_param('Calibration parameters,R-square:%.4f,Absolute space distance:%.4e,Absolute diff distance:%.4e' %
                   (r_square, abs_space_distance, abs_diff_distance))
    return poly, curve_cluster, all_calibration_curve, control_status_num, calibration_curve, cali_control, r_square, abs_space_distance, abs_diff_distance


def get_noise_overall(ori_data, cali_data):
    """输入标定后的曲线和原始数据，返回排序后的噪声(原始值-标定值) 认为每一种离散状态都是相同的分布"""
    device_status_num = ori_data.shape[0]  # 器件状态数
    curve_num = ori_data.shape[1]  # 曲线数目
    sorted_noise = np.zeros(device_status_num * curve_num)
    for i in range(curve_num):
        sorted_noise[i * device_status_num:(i + 1) * device_status_num] = ori_data[:, i] - cali_data
    sorted_noise = np.sort(sorted_noise, kind='mergesort')
    noise_min = sorted_noise.min()
    noise_max = sorted_noise.max()
    noise_mean = sorted_noise.mean()
    noise_var = sorted_noise.var()
    # ut.print_param('Noise statistics,min:%.4e max:%.4e mean:%.4e var:%.4e'
    #                % (noise_min, noise_max, noise_mean, noise_var))
    return sorted_noise, noise_min, noise_max, noise_mean, noise_var


def gen_noise_overall(sorted_noise, var_size, var_dim, var_type):
    """根据排序后的噪声得到同分布的噪声，大小指定,类型指定 适用于假定分布与状态无关 适用于偏置型量化"""
    """size:N C H W"""
    noise_num = sorted_noise.size  # 样本噪声个数
    if var_dim == 4:
        rand_p = np.random.rand(var_size[0], var_size[1], var_size[2], var_size[3]) * (noise_num - 1)  # 获得随机数池 0~n-1
    elif var_dim == 3:
        rand_p = np.random.rand(var_size[0], var_size[1], var_size[2]) * (noise_num - 1)  # 获得随机数池 0~n-1
    elif var_dim == 2:
        rand_p = np.random.rand(var_size[0], var_size[1]) * (noise_num - 1)  # 获得随机数池 0~n-1
    elif var_dim == 1:
        rand_p = np.random.rand(var_size[0]) * (noise_num - 1)  # 获得随机数池 0~n-1
    else:
        rand_p = 0
        ut.print_error('Error var size', current_name)
    index = np.fix(rand_p).astype(int)  # 得到下标矩阵 0~n-2 (排序噪声下标0~n-1)
    generated_noise = sorted_noise[index] + (rand_p - np.fix(rand_p)) * (sorted_noise[index + 1] - sorted_noise[index])
    if var_type == 'numpy':
        return generated_noise
    elif var_type == 'tensor':
        generated_noise = torch.from_numpy(generated_noise)
        return generated_noise
    else:
        ut.print_warn('Unknown variable type,return type is allocated as numpy array', current_name)
        return generated_noise


def get_noise_perstatus(ori_data, cali_data):
    """输入标定后的曲线和原始数据，返回排序后的噪声(原始值-标定值) 认为每一种离散状态都是不同的分布"""
    device_status_num = ori_data.shape[0]  # 器件状态数
    curve_num = ori_data.shape[1]  # 曲线数目
    sorted_noise = np.zeros((device_status_num, curve_num))  # 每一种器件状态数由curve_num个小噪声样本
    for i in range(device_status_num):
        sorted_noise[i, :] = np.sort(ori_data[i, :] - cali_data[i], kind='mergesort')  # 测量噪声以及排序
    noise_min = sorted_noise.min()
    noise_max = sorted_noise.max()
    noise_mean = sorted_noise.mean()
    noise_var = sorted_noise.var()
    # ut.print_param('Noise statistics,min:%.4e max:%.4e mean:%.4e var:%.4e'
    #                % (noise_min, noise_max, noise_mean, noise_var))
    return sorted_noise, noise_min, noise_max, noise_mean, noise_var


def gen_noise_perstatus(sorted_noise_overall, weight_tag, var_size, var_dim, var_type):
    """根据排序后的噪声得到同分布的噪声，大小指定,类型指定 适用于假定分布与状态有关 适用于偏置型"""
    """size:N C H W"""
    noise_num = sorted_noise_overall.shape[0]  # 样本噪声个数(有几类不同分布的噪声)
    noise_volume = sorted_noise_overall.shape[1]  # 样本噪声容量
    if var_dim == 4:
        tensor_size = (var_size[0], var_size[1], var_size[2], var_size[3])
    elif var_dim == 3:
        tensor_size = (var_size[0], var_size[1], var_size[2])
    elif var_dim == 2:
        tensor_size = (var_size[0], var_size[1])
    elif var_dim == 1:
        tensor_size = (var_size[0])
    else:
        tensor_size = 0
        ut.print_error('Error var size', current_name)
    generated_noise = np.zeros(tensor_size)  # 最终产生的噪声张量
    for i in range(noise_num):
        rand_p = np.random.rand(*tensor_size) * (noise_volume - 1)  # 获得随机数池 0~n-1 每一个不同的分布用不同的随机数池
        sorted_noise = np.squeeze(sorted_noise_overall[i, :])  # 得到噪声切片 即参考样本
        index = np.fix(rand_p).astype(int)  # 得到下标矩阵 0~n-2 (排序噪声下标0~n-1)
        tmp_noise = sorted_noise[index] + (rand_p - np.fix(rand_p)) * (sorted_noise[index + 1] - sorted_noise[index])
        tmp_noise[weight_tag != i] = 0  # 将不是这一个状态的噪声全部置零
        generated_noise = generated_noise + tmp_noise  # 更新输出噪声张量
    if var_type == 'numpy':
        return generated_noise
    elif var_type == 'tensor':
        generated_noise = torch.from_numpy(generated_noise)
        return generated_noise
    else:
        ut.print_warn('Unknown variable type,return type is allocated as numpy array', current_name)
        return generated_noise


def device_broken_fold(weight_tensor, ratio, var_size, var_dim, broken_type):
    """器件损坏模拟 适用于折叠架构
    weight_tensor:cpu tensor
    """
    if var_dim == 4:
        tensor_size = (var_size[0], var_size[1], var_size[2], var_size[3])
    elif var_dim == 3:
        tensor_size = (var_size[0], var_size[1], var_size[2])
    elif var_dim == 2:
        tensor_size = (var_size[0], var_size[1])
    elif var_dim == 1:
        tensor_size = (var_size[0])
    else:
        tensor_size = 0
        ut.print_error('Error var size', current_name)
    rand = torch.rand(*tensor_size)  # 0~1与输入张量尺寸一致的随机数
    out_tensor = copy.deepcopy(weight_tensor)
    if broken_type == 'ForceZeros':
        out_tensor[rand < ratio] = 0
        return out_tensor
    elif broken_type == 'Random':
        max_w = torch.max(weight_tensor)
        min_w = torch.min(weight_tensor)
        num = torch.sum((rand < ratio))
        out_tensor[rand < ratio] = torch.rand(num) * (max_w - min_w) + min_w
        return out_tensor
    elif broken_type == 'Fail':
        tmp = copy.deepcopy(out_tensor[rand < ratio])
        out_tensor[rand < ratio] = tmp * 0.3
        return out_tensor


def device_broken_unroll(tensor_set, ratio, broken_type, device):
    """器件损坏模拟 适用于展开架构
    tensor_set:weight tensor sets
    """
    var_size = tensor_set[0, 0].shape
    var_dim = tensor_set[0, 0].ndim
    if var_dim == 4:
        tensor_size = (var_size[0], var_size[1], var_size[2], var_size[3])
    elif var_dim == 3:
        tensor_size = (var_size[0], var_size[1], var_size[2])
    elif var_dim == 2:
        tensor_size = (var_size[0], var_size[1])
    elif var_dim == 1:
        tensor_size = (var_size[0])
    else:
        tensor_size = 0
        ut.print_error('Error var size', current_name)
    row = tensor_set.shape[0]
    col = tensor_set.shape[1]
    out_tensor_set = np.empty((row, col), dtype=object)
    if broken_type == 'ForceZeros':
        for i in range(row):
            for j in range(col):
                rand = torch.rand(*tensor_size)  # 0~1与输入张量尺寸一致的随机数
                weight = copy.deepcopy(tensor_set[i, j].cpu())  # 取切片分量
                weight[rand < ratio] = 0
                out_tensor_set[i, j] = copy.deepcopy(weight.to(device))
        return out_tensor_set
    elif broken_type == 'Random':
        for i in range(row):
            for j in range(col):
                rand = torch.rand(*tensor_size)  # 0~1与输入张量尺寸一致的随机数
                weight = copy.deepcopy(tensor_set[i, j].cpu())  # 取切片分量
                max_w = torch.max(weight)
                min_w = torch.min(weight)
                num = torch.sum(rand < ratio)
                weight[rand < ratio] = torch.rand(num) * (max_w - min_w) + min_w
                out_tensor_set[i, j] = copy.deepcopy(weight.to(device))
        return out_tensor_set
    elif broken_type == 'Fail':
        for i in range(row):
            for j in range(col):
                rand = torch.rand(*tensor_size)  # 0~1与输入张量尺寸一致的随机数
                weight = copy.deepcopy(tensor_set[i, j].cpu())  # 取切片分量
                tmp = copy.deepcopy(weight[rand < ratio])
                weight[rand < ratio] = tmp * 0.3
                out_tensor_set[i, j] = copy.deepcopy(weight.to(device))
        return out_tensor_set


def voltage_time(time):
    """
    :param time: 消逝时间
    :return: drop_scale根据拟合曲线返回在经过输入时间后电压下降系数
    """
    if time > 0:
        drop_scale = -0.001169 * time ** 0.513 + 1
    else:
        drop_scale = 1
    return drop_scale


def conductance_voltage(voltage):
    """
    :param voltage: 输入电压 在这里3V时 scale = 1
    :return: 根据输入电压
    """
    if voltage > 0:
        drop_scale = 0.028019888856391 * voltage ** 3.254
    else:
        drop_scale = 1
    return drop_scale
