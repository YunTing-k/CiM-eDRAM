"""器件权重映射等相关的函数和参数定义"""
import os
import Utilities as ut
import GlobalParametersManager as gpm
import numpy as np
import torch
import math

current_name = os.path.basename(__file__)  # 当前模块名字


def set_default_param():
    """设置默认参数"""
    """
    MNIST:0.98
    FashionMNIST:0.99
    notMNIST:0.98
    """
    if gpm.get_param('Train_dataset_type') == 'MNIST':
        gpm.set_param('Quant_truncate_ratio', 0.98)  # 非0元素的量化裁剪比例
    if gpm.get_param('Train_dataset_type') == 'FashionMNIST':
        gpm.set_param('Quant_truncate_ratio', 0.98)  # 非0元素的量化裁剪比例
    if gpm.get_param('Train_dataset_type') == 'notMNIST':
        gpm.set_param('Quant_truncate_ratio', 0.98)  # 非0元素的量化裁剪比例
    if gpm.get_param('Train_dataset_type') == 'CIFAR10':
        gpm.set_param('Quant_truncate_ratio', 0.99)  # 非0元素的量化裁剪比例
    gpm.set_param('Quantization_Method', 'global')  # 量化方法 DTCO方法，每一层都用DTCO的device设计
    # gpm.set_param('Quantization_Method', 'single')  # 量化方法 每一层用1：2：4：8的设计
    gpm.set_param('Quant_precision', 1e-15)  # 量化精度
    gpm.set_param('Quant_type', 'Diff')  # 器件量化类型 Diff-差分 Bias-偏置不差分
    gpm.set_param('Bias_Force_Zero', False)  # Bias量化时 是否强制让0点归零 对噪声添加也有影响 可强制让噪声为0
    gpm.set_param('Diff_Force_Zero', False)  # Diff量化时 是否强制让0点归零 对噪声添加也有影响 可强制让噪声为0
    """要启用Diff_Zero_Method的off 必须配合Diff_Force_Zero = True"""
    gpm.set_param('Diff_Zero_Method', 'min')  # Diff量化时 0权重的映射方式 min:minPhys - minPhys off:0 - 0 any:无任何修改
    gpm.set_param('Diff_Enhanced_Power_Efficiency', False)  # 在理想差分量化时 总是选择最小的物理组合作为量化结果
    gpm.set_param('Diff_Unique', False)  # 差分阵列是否剔除重复元素
    gpm.set_param('Prune_ratio', (0.5, 0.9))  # (conv,fc) 剪枝比例
    ut.print_info('Model deployment parameters added', current_name)


def quantize_model(input_tensor, quantize_lut, quantize_status_num, quantize_precision):
    """根据量化列表进行参数量化"""
    input_size = input_tensor.shape  # 输入张量大小
    """默认是最小的"""
    quantized_tensor_index = np.zeros(input_size).astype(int)  # 输出的量化权重所对应的器件值索引 0~n-1 初始化 -1
    diff_old = input_tensor - quantize_lut[0]  # 输入张量与第一个量化值做差
    for i in range(1, quantize_status_num):  # 从1开始到 quantize_status_num-1
        diff_new = input_tensor - quantize_lut[i]  # 输入张量与第i个量化值做差
        quantized_tensor_index[abs(diff_new) <= quantize_precision] = i  # 小于接受精度的情况下判定为相等
        mul = (diff_new < 0) ^ (diff_old < 0)  # mul仅在new<0 old>=0 以及new>=0 old<0为true 以异或运算代替乘积判断异号
        diff_diff = abs(diff_new - diff_old)
        quantized_tensor_index[(mul == 1) & (diff_diff > 0)] = i - 1  # 差乘积异号 且新的差更大
        quantized_tensor_index[(mul == 1) & (diff_diff <= 0)] = i  # 差乘积异号 且新的差更小
        diff_old = diff_new
    diff_new = input_tensor - quantize_lut[quantize_status_num - 1]  # 输入张量与最大的量化值做差
    quantized_tensor_index[diff_new >= 0] = quantize_status_num - 1  # 比最大的量化值还大的时候取最大值
    quantized_tensor = quantize_lut[quantized_tensor_index]  # 根据索引得到量化输出
    quantized_tensor = torch.from_numpy(quantized_tensor)  # 转化为张量
    return quantized_tensor, quantized_tensor_index


def get_quant_interval(data, in_ratio, num_lim):
    """根据比例确定量化区间 data需要先用torch.cpu传输到到cpu上"""
    if in_ratio > 1:
        ratio = 0.9
        ut.print_warn('Wrong ratio(>1)', current_name)
    elif in_ratio < 0:
        ratio = 0.1
        ut.print_warn('Wrong ratio(<0)', current_name)
    else:
        ratio = in_ratio
    data_flatten = data.view(1, -1).numpy()
    data_nonzero = data_flatten[data_flatten != 0]
    interval = np.zeros((1, 2))
    sorted_data = np.sort(data_nonzero, kind='mergesort')  # 排序后的数组
    data_num = sorted_data.size
    if data_num <= num_lim:
        interval[0, 0] = np.min(sorted_data)
        interval[0, 1] = np.max(sorted_data)
    else:
        tag1 = math.trunc(data_num * (1 - ratio) / 2)
        tag2 = math.trunc(data_num - data_num * (1 - ratio) / 2)
        if tag2 == data_num:
            tag2 = data_num - 1
        interval[0, 0] = sorted_data[tag1]
        interval[0, 1] = sorted_data[tag2]
    return interval


def get_range_channel(data, channel, method):
    """根据张量和通道得到对应的范围值 是weight equalization的一部分方法"""
    if method == 'Average':
        if data.ndim == 4:
            data_slice = data[channel, :, :, :]
        elif data.ndim == 2:
            data_slice = data[channel, :]
        weight_range = torch.abs(torch.min(data_slice)) / 2 + torch.abs(torch.max(data_slice)) / 2
        return weight_range
    elif method == 'Range':
        if data.ndim == 4:
            data_slice = data[channel, :, :, :]
        elif data.ndim == 2:
            data_slice = data[channel, :]
        weight_range = torch.max(data_slice) - torch.min(data_slice)
        return weight_range
    elif method == 'Max':
        if data.ndim == 4:
            data_slice = data[channel, :, :, :]
        elif data.ndim == 2:
            data_slice = data[channel, :]
        weight_range = torch.max(torch.abs(data_slice))
        return weight_range


def get_range(data, method):
    """根据张量得到对应的范围值 是weight equalization的一部分方法"""
    if method == 'Average':
        if data.ndim == 4:
            r_min = torch.amin(data, dim=(1, 2, 3))
            r_max = torch.amax(data, dim=(1, 2, 3))
        else:
            r_min = torch.amin(data, dim=1)
            r_max = torch.amax(data, dim=1)
        return (torch.abs(r_max) + torch.abs(r_min)) / 2
    elif method == 'Range':
        if data.ndim == 4:
            r_min = torch.amin(data, dim=(1, 2, 3))
            r_max = torch.amax(data, dim=(1, 2, 3))
        else:
            r_min = torch.amin(data, dim=1)
            r_max = torch.amax(data, dim=1)
        return r_max - r_min
    elif method == 'Max':
        if data.ndim == 4:
            r_max = torch.amax(torch.abs(data), dim=(1, 2, 3))
        else:
            r_max = torch.amax(torch.abs(data), dim=1)
        return r_max


def get_diff_lut(cali_data, if_unique):
    """根据原始标定数据得出差分后的标定数据 以及对应的W+ tag以及W-t ag
    最后返回的ctrl_plus和ctrl_minus是对应的原始序数 而不是原始物理量"""
    ctrl_tag = np.argsort(np.squeeze(cali_data), kind='mergesort')  # 排序后的下标
    cali_weight = np.sort(cali_data, kind='mergesort')  # 排序后的标定值
    ele_num = cali_data.shape[0]
    ctrl_tag_plus = np.zeros([ele_num ** 2, 1])
    ctrl_tag_minus = np.zeros([ele_num ** 2, 1])
    response_look_up = np.zeros([ele_num ** 2, 1])
    num = 0
    for i in range(ele_num):
        for j in range(ele_num):
            response_look_up[num, 0] = cali_weight[i] - cali_weight[j]
            ctrl_tag_plus[num, 0] = ctrl_tag[i]  # 记录正部分的下标
            ctrl_tag_minus[num, 0] = ctrl_tag[j]  # 记录负部分的下标
            num = num + 1
    if if_unique:  # 需要剔除一样的元素
        response_look_up, unique_tag = np.unique(response_look_up, return_index=True)  # lut已经被唯一化且排序了
        ctrl_plus = ctrl_tag_plus[unique_tag].astype(np.int)  # 最终对应下标
        ctrl_minus = ctrl_tag_minus[unique_tag].astype(np.int)  # 最终对应下标
        status_num = response_look_up.shape[0]  # 取得唯一元素的数目
        return response_look_up, ctrl_plus, ctrl_minus, status_num
    else:  # 不需要剔除一样的元素
        _plus_tag = np.argsort(response_look_up[:, 0], kind='mergesort')
        _minus_tag = np.argsort(response_look_up[:, 0], kind='mergesort')
        ctrl_plus = ctrl_tag_plus[_plus_tag].astype(np.int)  # 最终对应下标
        ctrl_minus = ctrl_tag_minus[_minus_tag].astype(np.int)  # 最终对应下标
        response_look_up = np.sort(response_look_up[:, 0], kind='mergesort')  # 最终排序后的lut
        status_num = ele_num ** 2
        return response_look_up, ctrl_plus, ctrl_minus, status_num


def get_diff_lut_enhanced(cali_data, area_lut):
    """根据原始标定数据得出差分后的标定数据 以及对应的W+ tag以及W-t ag
    最后返回的ctrl_plus和ctrl_minus是对应的原始序数 而不是原始物理量
    并且返回的权重一定是唯一的，并且总是具有与area lut对应的最小值"""
    ctrl_tag = np.argsort(np.squeeze(cali_data), kind='mergesort')  # 排序后的下标
    cali_weight = np.sort(cali_data, kind='mergesort')  # 排序后的标定值
    ele_num = cali_data.shape[0]
    ctrl_tag_plus = np.zeros([ele_num ** 2, 1])
    ctrl_tag_minus = np.zeros([ele_num ** 2, 1])
    response_look_up = np.zeros([ele_num ** 2, 1])
    area_ref = np.zeros([ele_num ** 2, 1])
    num = 0
    for i in range(ele_num):
        for j in range(ele_num):
            response_look_up[num, 0] = cali_weight[i] - cali_weight[j]
            ctrl_tag_plus[num, 0] = ctrl_tag[i]  # 记录正部分的下标
            ctrl_tag_minus[num, 0] = ctrl_tag[j]  # 记录负部分的下标
            area_ref[num] = area_lut[i] + area_lut[j]  # 差分对应的area大小
            num = num + 1
    lut_unique = unique_floats(response_look_up, 5e-7)  # lut唯一化
    lut_unique = lut_unique[:, 0]
    lut_unique_sorted = np.sort(lut_unique, kind='mergesort')  # 排序
    unique_tag = np.zeros(np.size(lut_unique))
    num = 0
    for ele in lut_unique_sorted:
        ele_tag = np.where(np.isclose(response_look_up, ele, rtol=0, atol=5e-7) == True)  # 与ele相同的下标
        area = area_ref[ele_tag]  # 对应的area大小
        ele_tag = ele_tag[0]
        unique_tag[num] = ele_tag[np.argmin(area)]
        num = num + 1
    ctrl_plus = ctrl_tag_plus[unique_tag.astype(np.int)].astype(np.int)  # 最终对应下标
    ctrl_minus = ctrl_tag_minus[unique_tag.astype(np.int)].astype(np.int)  # 最终对应下标
    status_num = lut_unique_sorted.shape[0]  # 取得唯一元素的数目
    print(status_num)
    return lut_unique_sorted, ctrl_plus, ctrl_minus, status_num


def unique_floats(a, delta=1e-9):
    unique_values = []
    for value in a:
        if not any(np.isclose(value, u, rtol=0, atol=delta) for u in unique_values):
            unique_values.append(value)
    return np.array(unique_values)


def normalize_curve(input_data, target_max, target_min):
    """将曲线归一化到max和min划分的区间内 a * b + c形式"""
    data_max = np.max(input_data)
    data_min = np.min(input_data)
    scale = (target_max - target_min) / (data_max - data_min)
    bias = (target_min * data_max - target_max * data_min) / (data_max - data_min)
    norm_data = input_data * scale + bias
    return norm_data, scale, bias


def scale_curve(input_data, target_max, target_min):
    """将曲线拉伸以覆盖选定区间 a * b形式"""
    data_max = np.max(input_data)
    data_min = np.min(input_data)
    scale1 = target_max / data_max
    scale2 = target_min / data_min
    if scale1 >= scale2:
        scale = scale1
    else:
        scale = scale2
    norm_data = input_data * scale
    return norm_data, scale
