"""main program"""
import copy
import os
import time
import torch
import winsound

import GlobalParametersManager as gpm
import HardNet
import NN_AddNoise
import NN_Quantization
import NN_Train
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
    """Program start"""
    ut.print_debug('Program start at' + time.asctime(), current_name)

    """Parameter initialization"""
    gpm.init()  # 构建全局参数字典
    SoftTrainParameters.set_default_param()  # 构建软网络训练参数
    SoftInferenceParameters.set_default_param()  # 构建软网络推理参数
    Device.set_default_param()  # 构建器件参数
    ModelDeployment.set_default_param()  # 模型压缩参数

    """Build network"""
    if get('Train_dataset_type') == 'MNIST':
        net = SoftNet.MNIST_Type1_Template().to(get('Train_device'))  # 构建网络
    elif get('Train_dataset_type') == 'FashionMNIST':
        net = SoftNet.FashionMNIST_Type1_Template().to(get('Train_device'))  # 构建网络
    elif get('Train_dataset_type') == 'notMNIST':
        net = SoftNet.notMNIST_Type1_Template().to(get('Train_device'))  # 构建网络
    elif get('Train_dataset_type') == 'CIFAR10':
        net = SoftNet.CIFAR10_Type1_Template().to(get('Train_device'))  # 构建网络
    else:
        net = SoftNet.MNIST_Type1_Template().to(get('Train_device'))  # 构建网络
        ut.print_warn('Undefined Train_dataset_type', current_name)
    ut.print_debug('Current dataset is ' + get('Train_dataset_type'), current_name)

    """Model training/loading"""
    if not get('Train_if_load_model'):
        """Train network"""
        net, _, _ = NN_Train.train(net)
        tmp_net_dict = copy.deepcopy(net.state_dict())  # 暂存模型数据
        torch.save(tmp_net_dict, './Model/' + gpm.get_param('Model_name') + '.pth')
        ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Original')
        ut.print_info('FP32-dense softnet model saved in /Model', current_name)
        """Inference"""
        NN_Inference.inference(net, False, 'Train')
        if get('TrainSet_Inference'):
            NN_Inference.inference(net, True, 'Train')
        """Build the network for pruning"""
        if get('Train_dataset_type') == 'MNIST':
            net = SoftNet.MNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'FashionMNIST':
            net = SoftNet.FashionMNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'notMNIST':
            net = SoftNet.notMNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'CIFAR10':
            net = SoftNet.CIFAR10_Type1_TemplateU().to(get('Train_device'))
        else:
            net = SoftNet.MNIST_Type1_TemplateU().to(get('Train_device'))
            ut.print_warn('Undefined Train_dataset_type', current_name)
        net.load_state_dict(tmp_net_dict)  # 从暂存数据中加载权值
    # We load the model from disk"""
    else:
        """Build network for pruning"""
        if get('Train_dataset_type') == 'MNIST':
            net = SoftNet.MNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'FashionMNIST':
            net = SoftNet.FashionMNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'notMNIST':
            net = SoftNet.notMNIST_Type1_TemplateU().to(get('Train_device'))
        elif get('Train_dataset_type') == 'CIFAR10':
            net = SoftNet.CIFAR10_Type1_TemplateU().to(get('Train_device'))
        else:
            net = SoftNet.MNIST_Type1_TemplateU().to(get('Train_device'))
            ut.print_warn('Undefined Train_dataset_type', current_name)
        """Load model"""
        net.load_state_dict(torch.load('./Model/' + gpm.get_param('Model_name') + '.pth'))
        ut.print_info('Softnet model loaded in /Model', current_name)
        if not get('Inference_skip'):
            NN_Inference.inference(net, False, 'Train')
            if get('TrainSet_Inference'):
                NN_Inference.inference(net, True, 'Train')
        else:
            ut.print_info('Inference is skipped', current_name)

    """Model pruning"""
    if get('If_prune'):
        if get('Prune_retrain'):
            net, _, _ = NN_Train.prun_train(net)
            SoftNet.get_sparsity(net.parameters())
            torch.save(net.state_dict(), './Model/Pruned' + gpm.get_param('Model_name') + '.pth')  # 保存剪枝后的模型
            ut.print_info('Pruned softnet model saved in /Model', current_name)
            ut.weight_to_mat(net.state_dict(), get('Train_dataset_type') + 'Pruned')
        else:
            net.load_state_dict(torch.load('./Model/Pruned' + gpm.get_param('Model_name') + '.pth'))  # 从保存文件中读取参数
            ut.print_info('Pruned softnet model loaded in /Model', current_name)
        """Post-pruning inference"""
        if not get('Prune_inference_skip'):
            NN_Inference.inference(net, False, 'Prune')
        else:
            ut.print_info('Inference after pruning is skipped', current_name)
    else:
        net.load_state_dict(torch.load('./Model/' + gpm.get_param('Model_name') + '.pth'))  # 从保存文件中读取参数
        ut.print_info('Original softnet model loaded in /Model', current_name)
        ut.print_info('Prune operation is skipped', current_name)

    """Quantization"""
    quantized_net = None
    if not get('Q_skip'):
        if not get('Q_Read'):
            # nonideal: variation aware 的量化，ideal: 根据理想器件响应的量化
            (quantized_net, quantized_weight_lut_sets, quantized_tag_lut_sets,
             quantized_tensor_tag, quant_scale_sets, quant_bias_sets) = \
                NN_Quantization.quantization(net, get('Device_File_ideal'), get('Quantization_Method'))
            if not get('Q_inference_skip'):
                NN_Inference.inference(quantized_net, False, 'Quant')
                if get('TrainSet_Inference'):
                    NN_Inference.inference(quantized_net, True, 'Quant')
            else:
                ut.print_info('Post-quantization Inference is skipped', current_name)
        else:
            """Read data"""
            if get('If_prune'):
                net.load_state_dict(
                    torch.load('./Model/' + 'Pruned+Quanted' + gpm.get_param('Model_name') + '.pth'))
            else:
                net.load_state_dict(torch.load('./Model/' + 'Quanted' + gpm.get_param('Model_name') + '.pth'))
            quantized_net = copy.deepcopy(net)
            if not get('Q_inference_skip'):
                NN_Inference.inference(quantized_net, False, 'Quant')
                if get('TrainSet_Inference'):
                    NN_Inference.inference(quantized_net, True, 'Quant')
            else:
                ut.print_info('Quantization inference is skipped', current_name)

    """Noise added simulation"""
    Noise_Method = get('Noise_Method')  # 噪声添加策略
    if get('Include_Noise') and (not get('Q_skip')):
        if get('Mapping_Strategy') == 'Fold':
            noise_net = NN_AddNoise.add_noise_fold(quantized_net, get('Quantization_Method'))
        # elif get('Mapping_Strategy') == 'Unroll':
        else:
            noise_net = NN_AddNoise.add_noise_unroll(quantized_net, get('Quantization_Method'))
        """Inference after adding noise"""
        NN_Inference.inference(noise_net, False, 'Quant')
    else:
        ut.print_info('Noise consideration is skipped', current_name)

    """Performance Estimation"""
    if get('Performance_estimate') and (not get('Q_skip')):
        refresh_rate = get('Refresh_rate')
        power_net = HardNet.power_net_prepare(quantized_net, get('Device_File_ideal'), get('Quantization_Method'))
        [total_ops, t_total, power_avg, ops_avg, ops_w_avg] = \
            NN_Inference.performance_inference(power_net, False, 'Quant')
        [total_cap_area, total_refresh_energy, total_refresh_time, all_area] = HardNet.capacitance_net(
            quantized_net, get('Device_File_area'), get('Quantization_Method'))
        area_efficiency = ops_avg / (all_area * 1e-6)
        inference_ops_w = ops_avg / (power_avg + total_refresh_energy * refresh_rate[0])
        training_ops_w = ops_avg / (power_avg + total_refresh_energy * refresh_rate[1])
        ut.print_param('  ----Area Scale/Power Scale: %s/%s' % (str(get('If_Area_Scale')), str(get('If_Power_Scale'))))
        ut.print_param('  ----Scale Factor Width: %.4f Length: %.4f'
                       % (get('Width_Scale_factor'), get('Length_Scale_Factor')))
        ut.print_param('  ----Total OPs: %d' % total_ops)
        ut.print_param('  ----Total Cap Area: %.6e um2' % total_cap_area)
        ut.print_param('  ----Possible Maximum Refresh Frequency: %.6e Hz' % (1 / total_refresh_time))
        ut.print_param('  ----Total Refresh Energy: %.6e J' % total_refresh_energy)
        ut.print_param('  ----Average Computing-Only Power: %.6e W' % power_avg)
        ut.print_param('  ----Average OPS: %.6e OP/S' % ops_avg)
        ut.print_param('  ----Total Area: %.6e um2' % all_area)
        ut.print_param('  ----Area Efficiency: %.6e TOPS/mm2' % (area_efficiency * 1e-12))
        if total_refresh_time > (1 / refresh_rate[1]):  # 目标刷新时间短于最短时间
            ut.print_warn('Target inference refresh frequency is too high', current_name)
        ut.print_param('  ----Average Inference Computing Power: %.4f μW, %.4f%% of total' %
                       (power_avg * 1e6, 100 * power_avg / (power_avg + total_refresh_energy * refresh_rate[0])))
        ut.print_param('  ----Average Inference Refreshing Power: %.4f μW, %.4f%% of total' %
                       (total_refresh_energy * refresh_rate[0] * 1e6, 100 * (total_refresh_energy * refresh_rate[0]) /
                        (power_avg + total_refresh_energy * refresh_rate[0])))
        ut.print_param('  ----Inference Energy Efficiency: %.6f TOPS/W' % (inference_ops_w * 1e-12))
        if total_refresh_time > (1 / refresh_rate[1]):  # 目标刷新时间短于最短时间
            ut.print_warn('Target training refresh frequency is too high', current_name)
        ut.print_param('  ----Training Energy Efficiency: %.6f TOPS/W' % (training_ops_w * 1e-12))
    else:
        ut.print_info('Performance estimation is skipped', current_name)

    """Save all parameters"""
    gpm.write_param()
    ut.print_debug('Program end at ' + time.asctime(), current_name)

    """Program end here"""
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
