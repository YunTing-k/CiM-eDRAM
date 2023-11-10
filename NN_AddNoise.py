import copy
import os
import sys
import time

import numpy as np
import torch

import Device
import HardNet
import Utilities as ut
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字


def add_noise_fold(in_net, mode):
    """add noise to quantized network in fold mode"""
    """Preparation"""
    train_device = get('Train_device')  # 指定训练设备
    Noise_Method = get('Noise_Method')
    Discrete_Calibration_order = get('Discrete_Calibration_order')  # 离散器件标定级数
    """Read data"""
    quantized_tag_lut_sets = np.load('./Parameters/quantized_tag_lut_sets.npy', allow_pickle=True).item()
    quantized_tensor_tag = np.load('./Parameters/quantized_tensor_tag.npy', allow_pickle=True).item()
    quant_scale_sets = np.load('./Parameters/quant_scale_sets.npy', allow_pickle=True)
    """count the layer"""
    net = copy.deepcopy(in_net)  # 深拷贝
    net.eval()  # 推理模式
    layer_name = []  # 层的名称
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layer_name.append(name)
        elif isinstance(module, torch.nn.Linear):
            layer_name.append(name)
    if mode == 'single':
        """We used the same distribution to quant NN"""
        cim_curve_cluster, _, _, _, _, _ = Device.discrete_calibrate_curve(
            get('Device_File_nonideal') + get('Train_dataset_type') +
            'Single.mat', Discrete_Calibration_order)  # 得到原始数据
        _, _, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
            get('Device_File_ideal') + get('Train_dataset_type') +
            'Single.mat', Discrete_Calibration_order)  # 得到理想数据
    current_layer = 0
    for net_params in net.parameters():  # 遍历网络层进行量化
        if mode == 'global':
            """
            In each layer, we quant NN layers independently,
            so we need modeling CIM device during each layer quantization
            """
            cim_curve_cluster, _, _, _, _, _ = Device.discrete_calibrate_curve(
                get('Device_File_nonideal') + get('Train_dataset_type') +
                str(current_layer) + '.mat', Discrete_Calibration_order)  # 得到原始数据
            _, _, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
                get('Device_File_ideal') + get('Train_dataset_type') +
                str(current_layer) + '.mat', Discrete_Calibration_order)  # 得到理想数据
        if get('Quant_type') == 'Bias':  # CIM 偏置型
            force_zeros = get('Bias_Force_Zero')  # 是否强制归零
            """偏置型不用考虑正负不同差分对的分别量化"""
            if Noise_Method == 'Overall':  # CIM 偏置型 认为分布与状态无关
                overall_noise_sorted, _, _, _, _ = \
                    Device.get_noise_overall(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                noise_tensor = Device.gen_noise_overall(overall_noise_sorted,
                                                        net_params.data.shape,
                                                        net_params.data.ndim, 'tensor') \
                               * quant_scale_sets[current_layer, :]
            # elif Noise_Method == 'PerStatus':  # CIM 偏置型 认为分布与状态有关
            else:
                perstatus_noise_sorted, _, _, _, _ = \
                    Device.get_noise_perstatus(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                quantized_tag = quantized_tag_lut_sets[layer_name[current_layer]]  # 得到量化查找表对应的原始Cali数据的下标
                q_tensor_tag = quantized_tag[quantized_tensor_tag[layer_name[current_layer]]]
                noise_tensor = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag,
                                                          net_params.data.shape, net_params.data.ndim, 'tensor') \
                               * quant_scale_sets[current_layer, :]
        else:
            force_zeros = get('Diff_Force_Zero')  # 是否强制归零
            """差分型需要考虑正负不同差分对的分别量化"""
            if Noise_Method == 'Overall':  # CIM 差分型 认为分布与状态无关
                overall_noise_sorted, _, _, _, _ = \
                    Device.get_noise_overall(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                noise_tensor_plus = Device.gen_noise_overall(overall_noise_sorted,
                                                             net_params.data.shape,
                                                             net_params.data.ndim, 'tensor') \
                                    * quant_scale_sets[current_layer, :]
                noise_tensor_minus = Device.gen_noise_overall(overall_noise_sorted,
                                                              net_params.data.shape,
                                                              net_params.data.ndim, 'tensor') \
                                     * quant_scale_sets[current_layer, :]
                noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
            # elif Noise_Method == 'PerStatus':  # CIM 差分型 认为分布与状态有关
            else:
                perstatus_noise_sorted, _, _, _, _ = \
                    Device.get_noise_perstatus(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                quantized_tag_plus = quantized_tag_lut_sets[
                    layer_name[current_layer] + 'plus']  # 得到量化查找表对应的原始Cali数据的下标
                quantized_tag_minus = quantized_tag_lut_sets[
                    layer_name[current_layer] + 'minus']  # 得到量化查找表对应的原始Cali数据的下标
                q_tensor_tag_plus = np.squeeze(quantized_tag_plus)[
                    quantized_tensor_tag[layer_name[current_layer]]]
                q_tensor_tag_minus = np.squeeze(quantized_tag_minus)[
                    quantized_tensor_tag[layer_name[current_layer]]]
                noise_tensor_plus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_plus,
                                                               net_params.data.shape, net_params.data.ndim,
                                                               'tensor') \
                                    * quant_scale_sets[current_layer, :]
                noise_tensor_minus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_minus,
                                                                net_params.data.shape, net_params.data.ndim,
                                                                'tensor') \
                                     * quant_scale_sets[current_layer, :]
                noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
        if force_zeros:  # 是否强制归零
            q_ref = net_params.data.cpu()
            noise_tensor[q_ref == 0] = 0
        """将噪声张量添加进入原始权重数据"""
        net_params.data = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
        current_layer += 1  # 下一层
    if mode == 'single':
        ut.print_info('Noise tensors with single-device quantization for folded NN  are added', current_name)
    elif mode == 'global':
        ut.print_info('Noise tensors with global quantization for folded NN are added', current_name)
    else:
        ut.print_error('Undefined quantization method', current_name)
        sys.exit(1)
    return net


def add_noise_unroll(net, mode):
    """add noise to quantized network in unroll mode"""
    """Preparation"""
    train_device = get('Train_device')  # 指定训练设备
    Noise_Method = get('Noise_Method')
    Discrete_Calibration_order = get('Discrete_Calibration_order')  # 离散器件标定级数
    """Read data"""
    quantized_tag_lut_sets = np.load('./Parameters/quantized_tag_lut_sets.npy', allow_pickle=True).item()
    quantized_tensor_tag = np.load('./Parameters/quantized_tensor_tag.npy', allow_pickle=True).item()
    quant_scale_sets = np.load('./Parameters/quant_scale_sets.npy', allow_pickle=True)
    """count the layer"""
    net.eval()  # 推理模式
    layer_name = []  # 层的名称
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            layer_name.append(name)
        elif isinstance(module, torch.nn.Linear):
            layer_name.append(name)
    """Network structure"""
    if get('Train_dataset_type') == 'MNIST':
        channel_conv1 = 3
        height_conv1 = 14
        width_conv1 = 14

        channel_conv2 = 24
        height_conv2 = 7
        width_conv2 = 7
    elif get('Train_dataset_type') == 'FashionMNIST':
        channel_conv1 = 3
        height_conv1 = 14
        width_conv1 = 14

        channel_conv2 = 64
        height_conv2 = 6
        width_conv2 = 6
    elif get('Train_dataset_type') == 'notMNIST':
        channel_conv1 = 3
        height_conv1 = 14
        width_conv1 = 14

        channel_conv2 = 48
        height_conv2 = 7
        width_conv2 = 7
    elif get('Train_dataset_type') == 'CIFAR10':
        channel_conv1 = 64
        height_conv1 = 16
        width_conv1 = 16

        channel_conv2 = 64
        height_conv2 = 7
        width_conv2 = 7
    else:
        ut.print_error('Undefined trainset', current_name)
        sys.exit(1)
    if mode == 'single':
        """We used the same distribution to quant NN"""
        cim_curve_cluster, _, _, _, _, _ = Device.discrete_calibrate_curve(
            get('Device_File_nonideal') + get('Train_dataset_type') +
            'Single.mat', Discrete_Calibration_order)  # 得到原始数据
        _, _, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
            get('Device_File_ideal') + get('Train_dataset_type') +
            'Single.mat', Discrete_Calibration_order)  # 得到理想数据
    current_layer = 0  # 从第0层开始
    for net_params in net.parameters():  # 遍历网络层进行量化
        if mode == 'global':
            """
            In each layer, we quant NN layers independently,
            so we need modeling CIM device during each layer quantization
            """
            cim_curve_cluster, _, _, _, _, _ = Device.discrete_calibrate_curve(
                get('Device_File_nonideal') + get('Train_dataset_type') +
                str(current_layer) + '.mat', Discrete_Calibration_order)  # 得到原始数据
            _, _, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
                get('Device_File_ideal') + get('Train_dataset_type') +
                str(current_layer) + '.mat', Discrete_Calibration_order)  # 得到理想数据
        if get('Quant_type') == 'Bias':  # CIM 偏置型
            force_zeros = get('Bias_Force_Zero')  # 是否强制归零
            """偏置型不用考虑正负不同差分对的分别量化"""
            if Noise_Method == 'Overall':  # CIM 偏置型 认为分布与状态无关
                overall_noise_sorted, _, _, _, _ = \
                    Device.get_noise_overall(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                if current_layer == 0:
                    """conv1w"""
                    conv1w = np.empty((channel_conv1, height_conv1 * width_conv1), dtype=object)
                    itr = 0
                    for height in range(height_conv1):
                        for width in range(width_conv1):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                    net_params.data.ndim, 'tensor') \
                                           * quant_scale_sets[current_layer, :]
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv1):
                                conv1w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer == 1:
                    """conv2w"""
                    conv2w = np.empty((channel_conv2, height_conv2 * width_conv2), dtype=object)
                    itr = 0
                    for height in range(height_conv2):
                        for width in range(width_conv2):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                    net_params.data.ndim, 'tensor') \
                                           * quant_scale_sets[current_layer, :]
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv2):
                                conv2w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer <= 3:
                    """fc1 and fc2"""
                    """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                    noise_tensor = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                            net_params.data.ndim, 'tensor') \
                                   * quant_scale_sets[current_layer, :]
                    if force_zeros:  # 是否强制归零
                        q_ref = net_params.data.cpu()
                        noise_tensor[q_ref == 0] = 0
                    if current_layer == 2:
                        fc1 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    else:
                        fc2 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
            elif Noise_Method == 'PerStatus':  # CIM 偏置型 认为分布与状态有关
                perstatus_noise_sorted, _, _, _, _ = \
                    Device.get_noise_perstatus(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                quantized_tag = quantized_tag_lut_sets[layer_name[current_layer]]  # 得到量化查找表对应的原始Cali数据的下标
                q_tensor_tag = quantized_tag[quantized_tensor_tag[layer_name[current_layer]]]
                if current_layer == 0:
                    """conv1w"""
                    conv1w = np.empty((channel_conv1, height_conv1 * width_conv1), dtype=object)
                    itr = 0
                    for height in range(height_conv1):
                        for width in range(width_conv1):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag,
                                                                      net_params.data.shape, net_params.data.ndim,
                                                                      'tensor') \
                                           * quant_scale_sets[current_layer, :]
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv1):
                                conv1w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer == 1:
                    """conv2w"""
                    conv2w = np.empty((channel_conv2, height_conv2 * width_conv2), dtype=object)
                    itr = 0
                    for height in range(height_conv2):
                        for width in range(width_conv2):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag,
                                                                      net_params.data.shape, net_params.data.ndim,
                                                                      'tensor') \
                                           * quant_scale_sets[current_layer, :]
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv2):
                                conv2w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer <= 3:
                    """fc1 and fc2"""
                    """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                    noise_tensor = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag,
                                                              net_params.data.shape, net_params.data.ndim, 'tensor') \
                                   * quant_scale_sets[current_layer, :]
                    if force_zeros:  # 是否强制归零
                        q_ref = net_params.data.cpu()
                        noise_tensor[q_ref == 0] = 0
                    if current_layer == 2:
                        fc1 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    else:
                        fc2 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
        elif get('Quant_type') == 'Diff':  # CIM 差分型
            force_zeros = get('Diff_Force_Zero')  # 是否强制归零
            """差分型需要考虑正负不同差分对的分别量化"""
            if Noise_Method == 'Overall':  # CIM 差分型 认为分布与状态无关
                overall_noise_sorted, _, _, _, _ = \
                    Device.get_noise_overall(cim_curve_cluster, cim_cali_curve)  # 得到噪声样本(还是在物理域)
                if current_layer == 0:
                    """conv1w"""
                    conv1w = np.empty((channel_conv1, height_conv1 * width_conv1), dtype=object)
                    itr = 0
                    for height in range(height_conv1):
                        for width in range(width_conv1):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor_plus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                         net_params.data.ndim, 'tensor') \
                                                * quant_scale_sets[current_layer, :]
                            noise_tensor_minus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                          net_params.data.ndim, 'tensor') \
                                                 * quant_scale_sets[current_layer, :]
                            noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv1):
                                conv1w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer == 1:
                    """conv2w"""
                    conv2w = np.empty((channel_conv2, height_conv2 * width_conv2), dtype=object)
                    itr = 0
                    for height in range(height_conv2):
                        for width in range(width_conv2):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor_plus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                         net_params.data.ndim, 'tensor') \
                                                * quant_scale_sets[current_layer, :]
                            noise_tensor_minus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                          net_params.data.ndim, 'tensor') \
                                                 * quant_scale_sets[current_layer, :]
                            noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv2):
                                conv2w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer >= 2:
                    """fc1 fc2 fc3"""
                    """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                    noise_tensor_plus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                 net_params.data.ndim, 'tensor') \
                                        * quant_scale_sets[current_layer, :]
                    noise_tensor_minus = Device.gen_noise_overall(overall_noise_sorted, net_params.data.shape,
                                                                  net_params.data.ndim, 'tensor') \
                                         * quant_scale_sets[current_layer, :]
                    noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                    if force_zeros:  # 是否强制归零
                        q_ref = net_params.data.cpu()
                        noise_tensor[q_ref == 0] = 0
                    if current_layer == 2:
                        fc1 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    elif current_layer == 3:
                        fc2 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    else:
                        fc3 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
            elif Noise_Method == 'PerStatus':  # CIM 差分型 认为分布与状态有关
                perstatus_noise_sorted, _, _, _, _ = Device.get_noise_perstatus(cim_curve_cluster, cim_cali_curve)
                quantized_tag_plus = quantized_tag_lut_sets[layer_name[current_layer] + 'plus']  # 得到量化查找表对应的原始Cali数据的下标
                quantized_tag_minus = quantized_tag_lut_sets[layer_name[current_layer] + 'minus']  # 得到量化查找表对应原始Cali数据下标
                q_tensor_tag_plus = np.squeeze(quantized_tag_plus)[quantized_tensor_tag[layer_name[current_layer]]]
                q_tensor_tag_minus = np.squeeze(quantized_tag_minus)[quantized_tensor_tag[layer_name[current_layer]]]
                if current_layer == 0:
                    """conv1w"""
                    conv1w = np.empty((channel_conv1, height_conv1 * width_conv1), dtype=object)
                    itr = 0
                    for height in range(height_conv1):
                        for width in range(width_conv1):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor_plus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_plus,
                                                                           net_params.data.shape, net_params.data.ndim,
                                                                           'tensor') \
                                                * quant_scale_sets[current_layer, :]
                            noise_tensor_minus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_minus,
                                                                            net_params.data.shape, net_params.data.ndim,
                                                                            'tensor') \
                                                 * quant_scale_sets[current_layer, :]
                            noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv1):
                                conv1w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer == 1:
                    """conv2w"""
                    conv2w = np.empty((channel_conv2, height_conv2 * width_conv2), dtype=object)
                    itr = 0
                    for height in range(height_conv2):
                        for width in range(width_conv2):
                            """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                            noise_tensor_plus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_plus,
                                                                           net_params.data.shape, net_params.data.ndim,
                                                                           'tensor') \
                                                * quant_scale_sets[current_layer, :]
                            noise_tensor_minus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_minus,
                                                                            net_params.data.shape, net_params.data.ndim,
                                                                            'tensor') \
                                                 * quant_scale_sets[current_layer, :]
                            noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                            if force_zeros:  # 是否强制归零
                                q_ref = net_params.data.cpu()
                                noise_tensor[q_ref == 0] = 0
                            # N C H W
                            tmp_w = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                            for channel in range(channel_conv2):
                                conv2w[channel, itr] = tmp_w[channel, :, :, :]  # 1 C H W
                            itr = itr + 1
                elif current_layer >= 2:
                    """fc1 fc2 fc3"""
                    """产生噪声张量 其中物理域到模型域转换需要乘以scale 但不论是偏置还是差分 只需要线性scale就好"""
                    noise_tensor_plus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_plus,
                                                                   net_params.data.shape, net_params.data.ndim,
                                                                   'tensor') \
                                        * quant_scale_sets[current_layer, :]
                    noise_tensor_minus = Device.gen_noise_perstatus(perstatus_noise_sorted, q_tensor_tag_minus,
                                                                    net_params.data.shape, net_params.data.ndim,
                                                                    'tensor') \
                                         * quant_scale_sets[current_layer, :]
                    noise_tensor = noise_tensor_plus - noise_tensor_minus  # 由于是diff模式 需要分别考虑噪声
                    if force_zeros:  # 是否强制归零
                        q_ref = net_params.data.cpu()
                        noise_tensor[q_ref == 0] = 0
                    if current_layer == 2:
                        fc1 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    elif current_layer == 3:
                        fc2 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
                    else:
                        fc3 = net_params.data + noise_tensor.type(torch.FloatTensor).to(train_device)
        current_layer += 1  # 下一层
    if mode == 'single':
        ut.print_info('Noise tensors with single-device quantization for unrolled NN  are added', current_name)
    elif mode == 'global':
        ut.print_info('Noise tensors with global quantization for unrolled NN are added', current_name)
    else:
        ut.print_error('Undefined quantization method', current_name)
        sys.exit(1)
    """构建硬件展开专用网络"""
    if get('Train_dataset_type') == 'MNIST':
        noise_net = HardNet.MNIST_Type1_Template_Unroll(conv1=conv1w, conv2=conv2w, fc1=fc1, fc2=fc2)
    elif get('Train_dataset_type') == 'FashionMNIST':
        noise_net = HardNet.FashionMNIST_Type1_Template_Unroll(conv1=conv1w, conv2=conv2w, fc1=fc1, fc2=fc2)
    elif get('Train_dataset_type') == 'notMNIST':
        noise_net = HardNet.notMNIST_Type1_Template_Unroll(conv1=conv1w, conv2=conv2w, fc1=fc1, fc2=fc2)
    elif get('Train_dataset_type') == 'CIFAR10':
        noise_net = HardNet.CIFAR10_Type1_Template_Unroll(conv1=conv1w, conv2=conv2w, fc1=fc1, fc2=fc2, fc3=fc3)
    else:
        ut.print_error('Undefined Train_dataset_type', current_name)
        sys.exit(1)
    return noise_net
