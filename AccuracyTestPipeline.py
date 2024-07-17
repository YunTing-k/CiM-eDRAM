"""
Under different variation and method
Test the quantized and noise-induced
network accuracy
"""
import copy
import os
import time
import torch
import numpy as np
import winsound

import GlobalParametersManager as gpm
import NN_AddNoise
import NN_Quantization
import NN_Inference
import SoftInferenceParameters
import SoftTrainParameters
import ModelDeployment
import Device
import SoftNet
import Utilities as ut
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字
if __name__ == '__main__':
    """Four Cases of noise_aware||noise_induced:
    [1]: F F, accuracy under ideal situation
    [2]: F T, accuracy under noise
    [3]: T T, with noise aware, test accuracy under noise 
    [4]: T F, meaningless, omit this case"""
    train_acc = False  # False = 不验证训练集准确率 True = 验证训练集准确率

    cell_optimize = True  # False = 不优化器件 True = 优化器件
    quant_map_optimize = True  # False = 不进行量化+映射优化 True = 进行量化+映射优化
    unroll = quant_map_optimize  # False = 不展开网络 True = 展开网络
    noise_aware = quant_map_optimize  # False = 用理想数据做量化 True = 用含噪声的数据做量化
    noise_induced = True  # False = 不包含噪声 True = 包含噪声
    repeat_num = 20  # 一种情况下测试多少次
    variation_set = ['1%', '2%', '5%', '10%', '15%', '20%', '26%', '30%']
    # variation_set = ['1%', '30%']
    variation_num = len(variation_set)
    target_dataset_name = ['MNIST', 'CIFAR10']
    # target_dataset_name = ['CIFAR10']
    target_num = len(target_dataset_name)
    if train_acc:
        acc_array = np.zeros((target_num, variation_num, repeat_num, 2))  # 准确度结果 [A]种网络 [B]种不一致性 [C]次重复 [D]个数据(测试集/训练集)
    else:
        acc_array = np.zeros((target_num, variation_num, repeat_num))  # 准确度结果 [A]种网络 [B]种不一致性 [C]次重复
    """Program start"""
    ut.print_debug('Program start at' + time.asctime(), current_name)

    """Parameter initialization"""
    gpm.init()  # 构建全局参数字典
    SoftTrainParameters.set_default_param()  # 构建软网络训练参数
    SoftInferenceParameters.set_default_param()  # 构建软网络推理参数
    Device.set_default_param()  # 构建器件参数
    gpm.set_param('Include_Noise', noise_induced)  # 是否考虑噪声
    ModelDeployment.set_default_param()  # 模型部署参数
    if cell_optimize:
        gpm.set_param('Quantization_Method', 'global')  # 量化方法 DTCO方法，每一层都用DTCO的device设计
    else:
        gpm.set_param('Quantization_Method', 'single')  # 量化方法 每一层用1：2：4：8的设计
    if unroll:
        gpm.set_param('Mapping_Strategy', 'Unroll')  # 映射方式 影响最后的噪声评估结果 Unroll:权重对应多个器件
    else:
        gpm.set_param('Mapping_Strategy', 'Fold')  # 映射方式 影响最后的噪声评估结果 Fold:认为权重和器件一一对应
    """Accuracy Pipeline"""
    for net_type in range(target_num):  # 遍历网络
        gpm.set_param('Train_dataset_type', target_dataset_name[net_type])
        gpm.set_param('Model_name', 'Custom_' + gpm.get_param('Train_dataset_type'))  # 模型名称
        gpm.set_param('Train_dataset_path', './Data/' + gpm.get_param('Train_dataset_type') + '.mat')  # 数据集路径
        if gpm.get_param('Train_dataset_type') == 'MNIST':
            gpm.set_param('Quant_truncate_ratio', 0.98)  # 非0元素的量化裁剪比例
        if gpm.get_param('Train_dataset_type') == 'notMNIST':
            gpm.set_param('Quant_truncate_ratio', 0.98)  # 非0元素的量化裁剪比例
        if gpm.get_param('Train_dataset_type') == 'CIFAR10':
            gpm.set_param('Quant_truncate_ratio', 0.99)  # 非0元素的量化裁剪比例
        """Build network"""
        if get('Train_dataset_type') == 'MNIST':
            net = copy.deepcopy(SoftNet.MNIST_Type1_TemplateU().to(get('Train_device')))  # 构建网络
            ut.print_debug('Current dataset is ' + get('Train_dataset_type'), current_name)
        elif get('Train_dataset_type') == 'notMNIST':
            net = copy.deepcopy(SoftNet.notMNIST_Type1_TemplateU().to(get('Train_device')))  # 构建网络
            ut.print_debug('Current dataset is ' + get('Train_dataset_type'), current_name)
        elif get('Train_dataset_type') == 'CIFAR10':
            net = copy.deepcopy(SoftNet.CIFAR10_Type1_TemplateU().to(get('Train_device')))  # 构建网络
            ut.print_debug('Current dataset is ' + get('Train_dataset_type'), current_name)
        else:
            net = copy.deepcopy(SoftNet.MNIST_Type1_TemplateU().to(get('Train_device')))  # 构建网络
            ut.print_warn('Undefined Train_dataset_type', current_name)
            ut.print_debug('Current dataset is ' + get('Train_dataset_type'), current_name)

        """Load model"""
        net.load_state_dict(torch.load('./Model/' + gpm.get_param('Model_name') + '.pth'))
        ut.print_info('Softnet model loaded in /Model', current_name)
        if not get('Inference_skip'):
            NN_Inference.inference(net, False, 'Train')
            if get('TrainSet_Inference'):
                NN_Inference.inference(net, True, 'Train')
        else:
            ut.print_info('Inference is skipped', current_name)

        """If noise is considered, multiple simulation is needed"""
        if noise_induced:
            """Traverse the variation"""
            for net_variation in range(variation_num):
                gpm.set_param('Device_File_nonideal', './Device/ZnO/4bit/linear/nonideal/' + variation_set[net_variation] + '/')
                gpm.set_param('Device_File_ideal', './Device/ZnO/4bit/linear/ideal/')  # 曲线器件响应数据的路径
                if noise_aware:
                    quantized_net, quantized_weight_lut_sets, quantized_tag_lut_sets, \
                    quantized_tensor_tag, quant_scale_sets, quant_bias_sets \
                        = NN_Quantization.quantization(net, get('Device_File_nonideal'), get('Quantization_Method'))
                else:
                    quantized_net, quantized_weight_lut_sets, quantized_tag_lut_sets, \
                    quantized_tensor_tag, quant_scale_sets, quant_bias_sets \
                        = NN_Quantization.quantization(net, get('Device_File_ideal'), get('Quantization_Method'))

                for itr in range(repeat_num):
                    """Noise added simulation"""
                    Noise_Method = get('Noise_Method')  # 噪声添加策略
                    if get('Mapping_Strategy') == 'Fold':
                        noise_net = NN_AddNoise.add_noise_fold(quantized_net, get('Quantization_Method'))
                    # elif get('Mapping_Strategy') == 'Unroll':
                    else:
                        noise_net = NN_AddNoise.add_noise_unroll(quantized_net, get('Quantization_Method'))

                    """Inference after adding noise"""
                    if train_acc:
                        class_correct_test, class_total_test = NN_Inference.inference(noise_net, False, 'Quant')
                        acc_array[net_type, net_variation, itr, 0] = 100 * sum(class_correct_test) / sum(class_total_test)
                        class_correct_train, class_total_train = NN_Inference.inference(noise_net, True, 'Quant')
                        acc_array[net_type, net_variation, itr, 1] = 100 * sum(class_correct_train) / sum(class_total_train)
                    else:
                        class_correct_test, class_total_test = NN_Inference.inference(noise_net, False, 'Quant')
                        acc_array[net_type, net_variation, itr] = 100 * sum(class_correct_test) / sum(class_total_test)
        else:  # since T F case is meaningless, we only consider ideal quantization
            """Quantization"""
            quantized_net, quantized_weight_lut_sets, quantized_tag_lut_sets,\
            quantized_tensor_tag, quant_scale_sets, quant_bias_sets\
                = NN_Quantization.quantization(net, get('Device_File_ideal'), get('Quantization_Method'))

            """Inference"""
            NN_Inference.inference(quantized_net, False, 'Quant')

    """Save all parameters"""
    gpm.write_param()

    """Program end here"""
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
    ut.numpy_to_mat(acc_array, 'ACC' + str(noise_aware) + str(noise_induced) + get('Mapping_Strategy')
                    + get('Quantization_Method'))
    ut.print_debug('Program end at ' + time.asctime(), current_name)


