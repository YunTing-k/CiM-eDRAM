import copy
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

import Device
import ModelDeployment
import SoftNet
import Utilities as ut
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字


def quantization(in_net, device_file, mode):
    """Hardware quantization mode: global = overall quantization, single = quant with single one model"""
    """Preparation"""
    train_device = get('Train_device')  # 指定训练设备
    inference_device = get('Inference_device')  # 指定推理设备
    Q_truncate_ratio = get('Quant_truncate_ratio')  # 量化截断比例
    Quant_precision = get('Quant_precision')  # 量化精度
    Quant_type = get('Quant_type')  # 量化类型
    Discrete_Calibration_order = get('Discrete_Calibration_order')  # 离散器件标定级数
    """count the layer"""
    net = copy.deepcopy(in_net)
    net.eval()  # 推理模式
    conv_num = 0  # 卷积层数目
    fc_num = 0  # 全连接层数目
    layer_name = []  # 层的名称
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_num += 1
            layer_name.append(name)
        elif isinstance(module, torch.nn.Linear):
            fc_num += 1
            layer_name.append(name)
    layer_num = conv_num + fc_num
    """Dataloader"""
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=None, train=False)
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode='Quant')
    """Quantization results"""
    quantized_weight_lut_sets = {}  # 最终量化的查找表
    quantized_tag_lut_sets = {}  # 最终量化的权值张量所对应的控制数/状态序数
    quantized_tensor_tag = {}  # 被量化的张量其对应的输入查找表的下标
    quant_scale_sets = np.zeros((layer_num, 1))  # 量化的缩放次数
    quant_bias_sets = np.zeros((layer_num, 1))  # 量化的偏置系数
    Diff_Zero_Method = get('Diff_Zero_Method')  # 差分量化时 是否将为0的权值量化策略
    if mode == 'single':
        cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
            device_file + get('Train_dataset_type') + 'Single.mat', Discrete_Calibration_order)  # 得到原始数据 查找表
        cim_min_tag = np.argmin(cim_cali_curve)  # CIM器件最小物理量的下标
        area_lut = ut.mat_to_numpy(get('Device_File_area') + get('Train_dataset_type') + 'Single_Area.mat', 'area')
    else:
        pass
    current_layer = 0
    """Traverse the net and quantization"""
    for net_params in net.parameters():  # 遍历网络层进行量化
        """
        In each layer, we quant NN layers independently,
        so we need modeling CIM device during each layer quantization
        """
        if mode == 'global':
            cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
                device_file + get('Train_dataset_type') + str(current_layer) + '.mat',
                Discrete_Calibration_order)  # 得到原始数据 查找表
            cim_min_tag = np.argmin(cim_cali_curve)  # CIM器件最小物理量的下标
            area_lut = ut.mat_to_numpy(get('Device_File_area') + get('Train_dataset_type')
                                       + str(current_layer) + '_Area.mat', 'area')
        else:
            pass
        if Quant_type == 'Bias':  # CIM 偏置型
            quant_range = ModelDeployment.get_quant_interval(net_params.data.cpu(), Q_truncate_ratio, cim_status_num)
            ut.print_param(
                'Layer-%d,quant range is %.4f~%.4f' % (current_layer + 1, quant_range[0, 0], quant_range[0, 1]))
            quantized_weight, quant_scale, quant_bias = ModelDeployment.normalize_curve(
                cim_cali_curve, quant_range[0, 1], quant_range[0, 0])  # 进行归一化 CIM
            quantized_tag = np.argsort(quantized_weight, kind='mergesort')  # 得到下标
            quantized_weight = np.sort(quantized_weight, kind='mergesort')  # 重新排序
            quant_num = quantized_weight.shape[0]  # 要量化的精度数目
            quantized_weight = np.squeeze(quantized_weight)
            quantized_weight_lut_sets[layer_name[current_layer]] = quantized_weight  # 查找表更新
            quantized_tag_lut_sets[layer_name[current_layer]] = quantized_tag  # 查找表对应的cali数据的TAG
            quant_scale_sets[current_layer, :] = quant_scale  # 得到量化scale
            quant_bias_sets[current_layer, :] = quant_bias  # 得到量化bias
            force_zeros = get('Bias_Force_Zero')  # 是否强制归零
        else:
            quant_range = ModelDeployment.get_quant_interval(net_params.data.cpu(), Q_truncate_ratio,
                                                             cim_status_num ** 2)
            ut.print_param(
                'Layer-%d,quant range is %.4f~%.4f' % (current_layer + 1, quant_range[0, 0], quant_range[0, 1]))
            # 先得到差分格式 得到的diff_weight已经排序了
            if not get('Diff_Enhanced_Power_Efficiency'):  # 普通模式 不考虑area
                diff_weight, quantized_tag_plus, quantized_tag_minus, quant_num =\
                    ModelDeployment.get_diff_lut(cim_cali_curve, get('Diff_Unique'))
            else:
                diff_weight, quantized_tag_plus, quantized_tag_minus, quant_num =\
                    ModelDeployment.get_diff_lut_enhanced(cim_cali_curve, area_lut)
            quantized_weight, quant_scale = ModelDeployment.scale_curve(
                diff_weight, quant_range[0, 1], quant_range[0, 0])  # 再进行scale up 覆盖区间 CIM
            quantized_weight = np.squeeze(quantized_weight)
            quantized_weight_lut_sets[layer_name[current_layer]] = quantized_weight  # 查找表更新
            if (Diff_Zero_Method == 'min') or (Diff_Zero_Method == 'off'):
                quantized_tag_plus[quantized_weight == 0] = cim_min_tag
                quantized_tag_minus[quantized_weight == 0] = cim_min_tag
            quantized_tag_lut_sets[layer_name[current_layer] + 'plus'] = quantized_tag_plus  # 查找表的cali数据的TAG
            quantized_tag_lut_sets[layer_name[current_layer] + 'minus'] = quantized_tag_minus  # 查找表的cali数据的TAG
            quant_scale_sets[current_layer, :] = quant_scale  # 得到量化scale
            force_zeros = get('Diff_Force_Zero')  # 是否强制归零
        """模型量化"""
        q_tensor, q_tensor_index = \
            ModelDeployment.quantize_model(net_params.data.cpu().numpy(), quantized_weight, quant_num,
                                           Quant_precision)
        quantized_tensor_tag[layer_name[current_layer]] = q_tensor_index
        """权值覆写"""
        if force_zeros:
            q_ref = net_params.data.cpu()
            q_tensor[q_ref == 0] = 0  # 强制0元素量化为0
        net_params.data = copy.deepcopy(q_tensor.type(torch.FloatTensor).to(train_device))
        current_layer += 1  # 下一层
    if mode == 'global':
        ut.print_debug('Global quantization finished at ' + time.asctime(), current_name)
    elif mode == 'single':
        ut.print_debug('Single quantization finished at ' + time.asctime(), current_name)
    else:
        ut.print_error('Undefined quantization method', current_name)
        sys.exit(1)
    """存储其他文件"""
    np.save('./Parameters/quantized_weight_lut_sets.npy', quantized_weight_lut_sets)
    np.save('./Parameters/quantized_tag_lut_sets.npy', quantized_tag_lut_sets)
    np.save('./Parameters/quantized_tensor_tag.npy', quantized_tensor_tag)
    np.save('./Parameters/quant_scale_sets.npy', quant_scale_sets)
    np.save('./Parameters/quant_bias_sets.npy', quant_bias_sets)
    if get('If_prune'):
        torch.save(net.state_dict(), './Model/Pruned+Quanted' + get('Model_name') + '.pth')  # 保存剪枝和量化后的模型
        ut.print_info('Pruned and quanted softnet model saved in /Model', current_name)
        # 导出为MAT文件
        ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Pruned_Quanted')
    else:
        torch.save(net.state_dict(), './Model/Quanted' + get('Model_name') + '.pth')  # 保存量化后的模型
        ut.print_info('Quanted softnet model saved in /Model', current_name)
        # 导出为MAT文件
        ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Quanted')

    return net, quantized_weight_lut_sets, quantized_tag_lut_sets, quantized_tensor_tag, quant_scale_sets, quant_bias_sets


def quantization_drop(in_net, device_file, mode, elapsed_time):
    """
    Hardware quantization mode: global = overall quantization, single = quant with single one model
    Meanwhile, voltage drop induced conductance drop is considered
    """
    """Preparation"""
    train_device = get('Train_device')  # 指定训练设备
    inference_device = get('Inference_device')  # 指定推理设备
    Q_truncate_ratio = get('Quant_truncate_ratio')  # 量化截断比例
    Quant_precision = get('Quant_precision')  # 量化精度
    Quant_type = get('Quant_type')  # 量化类型
    Discrete_Calibration_order = get('Discrete_Calibration_order')  # 离散器件标定级数
    scale_factor = Device.conductance_voltage(3 * Device.voltage_time(elapsed_time))  # 由于retention引起的权重衰减
    """count the layer"""
    net = copy.deepcopy(in_net)
    net.eval()  # 推理模式
    conv_num = 0  # 卷积层数目
    fc_num = 0  # 全连接层数目
    layer_name = []  # 层的名称
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_num += 1
            layer_name.append(name)
        elif isinstance(module, torch.nn.Linear):
            fc_num += 1
            layer_name.append(name)
    layer_num = conv_num + fc_num
    """Dataloader"""
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=None, train=False)
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode='Quant')
    """Quantization results"""
    quantized_weight_lut_sets = {}  # 最终量化的查找表
    quantized_tag_lut_sets = {}  # 最终量化的权值张量所对应的控制数/状态序数
    quantized_tensor_tag = {}  # 被量化的张量其对应的输入查找表的下标
    quant_scale_sets = np.zeros((layer_num, 1))  # 量化的缩放次数
    quant_bias_sets = np.zeros((layer_num, 1))  # 量化的偏置系数
    Diff_Zero_Method = get('Diff_Zero_Method')  # 差分量化时 是否将为0的权值量化策略
    if mode == 'single':
        cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
            device_file + get('Train_dataset_type') + 'Single.mat', Discrete_Calibration_order)  # 得到原始数据 查找表
        cim_min_tag = np.argmin(cim_cali_curve)  # CIM器件最小物理量的下标
    else:
        pass
    current_layer = 0
    """Traverse the net and quantization"""
    for net_params in net.parameters():  # 遍历网络层进行量化
        """
        In each layer, we quant NN layers independently,
        so we need modeling CIM device during each layer quantization
        """
        if mode == 'global':
            cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
                device_file + get('Train_dataset_type') + str(current_layer) + '.mat',
                Discrete_Calibration_order)  # 得到原始数据 查找表
            cim_min_tag = np.argmin(cim_cali_curve)  # CIM器件最小物理量的下标
        else:
            pass
        if Quant_type == 'Bias':  # CIM 偏置型
            quant_range = ModelDeployment.get_quant_interval(net_params.data.cpu(), Q_truncate_ratio, cim_status_num)
            ut.print_param(
                'Layer-%d,quant range is %.4f~%.4f' % (current_layer + 1, quant_range[0, 0], quant_range[0, 1]))
            quantized_weight, quant_scale, quant_bias = ModelDeployment.normalize_curve(
                cim_cali_curve, quant_range[0, 1], quant_range[0, 0])  # 进行归一化 CIM
            quantized_tag = np.argsort(quantized_weight, kind='mergesort')  # 得到下标
            quantized_weight = np.sort(quantized_weight, kind='mergesort')  # 重新排序
            quant_num = quantized_weight.shape[0]  # 要量化的精度数目
            quantized_weight = np.squeeze(quantized_weight)
            quantized_weight_lut_sets[layer_name[current_layer]] = quantized_weight  # 查找表更新
            quantized_tag_lut_sets[layer_name[current_layer]] = quantized_tag  # 查找表对应的cali数据的TAG
            quant_scale_sets[current_layer, :] = quant_scale  # 得到量化scale
            quant_bias_sets[current_layer, :] = quant_bias  # 得到量化bias
            force_zeros = get('Bias_Force_Zero')  # 是否强制归零
        else:
            quant_range = ModelDeployment.get_quant_interval(net_params.data.cpu(), Q_truncate_ratio,
                                                             cim_status_num ** 2)
            ut.print_param(
                'Layer-%d,quant range is %.4f~%.4f' % (current_layer + 1, quant_range[0, 0], quant_range[0, 1]))
            # 先得到差分格式 得到的diff_weight已经排序了
            diff_weight, quantized_tag_plus, quantized_tag_minus, quant_num = \
                ModelDeployment.get_diff_lut(cim_cali_curve, get('Diff_Unique'))
            quantized_weight, quant_scale = ModelDeployment.scale_curve(
                diff_weight, quant_range[0, 1], quant_range[0, 0])  # 再进行scale up 覆盖区间 CIM
            quantized_weight = np.squeeze(quantized_weight)
            quantized_weight_lut_sets[layer_name[current_layer]] = quantized_weight  # 查找表更新
            if (Diff_Zero_Method == 'min') or (Diff_Zero_Method == 'off'):
                quantized_tag_plus[quantized_weight == 0] = cim_min_tag
                quantized_tag_minus[quantized_weight == 0] = cim_min_tag
            quantized_tag_lut_sets[layer_name[current_layer] + 'plus'] = quantized_tag_plus  # 查找表的cali数据的TAG
            quantized_tag_lut_sets[layer_name[current_layer] + 'minus'] = quantized_tag_minus  # 查找表的cali数据的TAG
            quant_scale_sets[current_layer, :] = quant_scale  # 得到量化scale
            force_zeros = get('Diff_Force_Zero')  # 是否强制归零
        """模型量化"""
        q_tensor, q_tensor_index = \
            ModelDeployment.quantize_model(net_params.data.cpu().numpy(), quantized_weight, quant_num,
                                           Quant_precision)
        quantized_tensor_tag[layer_name[current_layer]] = q_tensor_index
        """权重衰减"""
        q_tensor = q_tensor * scale_factor
        """权值覆写"""
        if force_zeros:
            q_ref = net_params.data.cpu()
            q_tensor[q_ref == 0] = 0  # 强制0元素量化为0
        net_params.data = copy.deepcopy(q_tensor.type(torch.FloatTensor).to(train_device))
        current_layer += 1  # 下一层
    if mode == 'global':
        ut.print_debug('Global quantization finished at ' + time.asctime(), current_name)
    elif mode == 'single':
        ut.print_debug('Single quantization finished at ' + time.asctime(), current_name)
    else:
        ut.print_error('Undefined quantization method', current_name)
        sys.exit(1)
    """存储其他文件"""
    np.save('./Parameters/quantized_weight_lut_sets.npy', quantized_weight_lut_sets)
    np.save('./Parameters/quantized_tag_lut_sets.npy', quantized_tag_lut_sets)
    np.save('./Parameters/quantized_tensor_tag.npy', quantized_tensor_tag)
    np.save('./Parameters/quant_scale_sets.npy', quant_scale_sets)
    np.save('./Parameters/quant_bias_sets.npy', quant_bias_sets)
    if get('If_prune'):
        torch.save(net.state_dict(), './Model/Pruned+Quanted' + get('Model_name') + '.pth')  # 保存剪枝和量化后的模型
        ut.print_info('Pruned and quanted softnet model saved in /Model', current_name)
        # 导出为MAT文件
        ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Pruned_Quanted')
    else:
        torch.save(net.state_dict(), './Model/Quanted' + get('Model_name') + '.pth')  # 保存量化后的模型
        ut.print_info('Quanted softnet model saved in /Model', current_name)
        # 导出为MAT文件
        ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Quanted')

    return net, quantized_weight_lut_sets, quantized_tag_lut_sets, quantized_tensor_tag, quant_scale_sets, quant_bias_sets
