import os
import sys
import time
import numpy
import torch
from tqdm import tqdm
import SoftNet
import Utilities as ut
import torchvision.transforms as transforms
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字


def inference(net, train, mode):
    """Inference of the input net"""
    """Preparation"""
    inference_device = get('Inference_device')  # 指定推理设备
    """Transform preparation"""
    if get('Train_dataset_type') == 'MNIST':
        custom_transformX = None
    elif get('Train_dataset_type') == 'FashionMNIST':
        custom_transformX = None
    elif get('Train_dataset_type') == 'notMNIST':
        custom_transformX = None
    elif get('Train_dataset_type') == 'CIFAR10':
        custom_transformX = None
        # custom_transformX = transforms.Compose([
        #     transforms.Normalize([0, 0, 0], [1, 1, 1])
        # ])
    else:
        custom_transformX = None
        ut.print_warn('Undefined Train_dataset_type', current_name)
    """Dataloader"""
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=custom_transformX, train=train)
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode=mode)

    ut.print_debug('Inference start at ' + time.asctime(), current_name)
    net.eval()  # 推理模式
    class_correct = list(0. for i in range(get('SoftNet_class_num')))
    class_total = list(0. for i in range(get('SoftNet_class_num')))
    process_bar = tqdm(enumerate(test_loader), total=len(test_loader),
                       desc='Soft Inference', leave=True, unit='batch', ncols=100)
    for index, (images, labels) in process_bar:
        with torch.no_grad():
            images = images.to(inference_device)
            labels = labels.to(inference_device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(get('Inference_batch')):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    process_bar.close()
    for i in range(get('SoftNet_class_num')):
        ut.print_param('Accuracy of Target type-%d : %.4f %%' % (i + 1, 100 * class_correct[i] / class_total[i]))
    ut.print_param('Overall accuracy : %.4f %%' % (100 * sum(class_correct) / sum(class_total)))
    return class_correct, class_total


def performance_inference(net, train, mode):
    """估计核心阵列性能"""
    """Preparation"""
    inference_device = get('Inference_device')  # 指定推理设备
    if get('Train_dataset_type') == 'MNIST':
        activity = [4 * 10 * 3 / (14 * 14 * 3 * 2 * 2),
                    12 * 10 * 4 / (7 * 7 * 24 * 2 * 2),
                    4 * 100 * 10 / (1176 * 512),
                    4 * 100 * 10 / (512 * 10)]
        t_pipeline_array = [(19.6 * 1 + 3) / get('Clock_freq'),
                            (4.9 * 6 + 4) / get('Clock_freq'),
                            (51.2 * 11.76 / 4 + 4) / get('Clock_freq'),
                            (5.12 / 4 + 4) / get('Clock_freq')]
        t_pipeline_cycles = [1, 1, 1, 1]
        # activity = [4 * 10 * 3 / (14 * 14 * 3 * 2 * 2),
        #             12 * 10 * 4 / (7 * 7 * 24 * 2 * 2 * 3),
        #             4 * 100 * 10 / (1176 * 512),
        #             4 * 100 * 10 / (512 * 10)]
        # t_pipeline_array = [(numpy.ceil(19.6) * 1) / get('Clock_freq'),
        #                     (numpy.ceil(4.9) * 6) / get('Clock_freq'),
        #                     (numpy.ceil(51.2) * numpy.ceil(11.76) / 4) / get('Clock_freq'),
        #                     (numpy.ceil(5.12) / 4) / get('Clock_freq')]
        # t_pipeline_cycles = [3 * 10, 4 * 10, 4 * 10, 4 * 10]
        total_ops = 14 * 14 * 3 * (2 * 2) + 7 * 7 * 24 * (2 * 2 * 3) + 1176 * 512 + 512 * 10
        custom_transformX = None
    elif get('Train_dataset_type') == 'CIFAR10':
        activity = [12 * 10 * 8 / (16 * 16 * 64 * 2 * 2),
                    100 * 10 * 8 / (7 * 7 * 64 * 2 * 2),
                    100 * 10 * 8 / (3136 * 1024),
                    100 * 10 * 8 / (1024 * 512),
                    100 * 10 * 4 / (512 * 10)]
        t_pipeline_array = [(8 * 25.6 + 4) / get('Clock_freq'),
                            (4.9 * 2.56 * 64 / 8 + 4) / get('Clock_freq'),
                            (102.4 * 31.36 / 8 + 4) / get('Clock_freq'),
                            (51.2 * 10.24 / 8 + 4) / get('Clock_freq'),
                            (5.12 / 4 + 4) / get('Clock_freq')]
        t_pipeline_cycles = [1, 1, 1, 1, 1]
        # activity = [12 * 10 * 8 / (16 * 16 * 64 * 2 * 2 * 3),
        #             100 * 10 * 8 / (7 * 7 * 64 * 2 * 2 * 64),
        #             100 * 10 * 8 / (3136 * 1024),
        #             100 * 10 * 8 / (1024 * 512),
        #             100 * 10 * 4 / (512 * 10)]
        # t_pipeline_array = [(8 * numpy.ceil(25.6)) / get('Clock_freq'),
        #                     (numpy.ceil(4.9) * numpy.ceil(2.56 * 64 / 8)) / get('Clock_freq'),
        #                     (numpy.ceil(102.4) * numpy.ceil(31.36 / 8)) / get('Clock_freq'),
        #                     (numpy.ceil(51.2) * numpy.ceil(10.24 / 8)) / get('Clock_freq'),
        #                     (numpy.ceil(5.12) / 4) / get('Clock_freq')]
        # t_pipeline_cycles = [1, 2, 3, 4, 5]
        total_ops = 16 * 16 * 64 * (2 * 2 * 3) + 7 * 7 * 64 * (2 * 2 * 64) + 3136 * 1024 + 1024 * 512 + 512 * 10
        custom_transformX = None
    else:
        ut.print_warn('Undefined Train_dataset_type', current_name)
        sys.exit(1)
    """Dataloader"""
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=custom_transformX, train=train)
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode=mode)

    ut.print_debug('Inference start at ' + time.asctime(), current_name)
    net.eval()  # 推理模式
    class_correct = list(0. for i in range(get('SoftNet_class_num')))
    class_total = list(0. for i in range(get('SoftNet_class_num')))
    process_bar = tqdm(enumerate(test_loader), total=len(test_loader),
                       desc='Soft Inference', leave=True, unit='batch', ncols=100)
    for index, (images, labels) in process_bar:
        with torch.no_grad():
            images = images.to(inference_device)
            labels = labels.to(inference_device)
            if get('Train_dataset_type') == 'MNIST':
                [p1, p2, p3, p4, outputs] = net(images)
            elif get('Train_dataset_type') == 'CIFAR10':
                [p1, p2, p3, p4, p5, outputs] = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(get('Inference_batch')):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    process_bar.close()
    for i in range(get('SoftNet_class_num')):
        ut.print_param('Accuracy of Target type-%d : %.4f %%' % (i + 1, 100 * class_correct[i] / class_total[i]))
    ut.print_param('Overall accuracy : %.4f %%' % (100 * sum(class_correct) / sum(class_total)))
    if get('Train_dataset_type') == 'MNIST':
        if get('If_Power_Scale'):
            power1 = get('Conductance_Scale_Factor') * p1 / sum(class_total)
            power2 = get('Conductance_Scale_Factor') * p2 / sum(class_total)
            power3 = get('Conductance_Scale_Factor') * p3 / sum(class_total)
            power4 = get('Conductance_Scale_Factor') * p4 / sum(class_total)
        else:
            power1 = p1 / sum(class_total)
            power2 = p2 / sum(class_total)
            power3 = p3 / sum(class_total)
            power4 = p4 / sum(class_total)
        t_total = (t_pipeline_array[0] * t_pipeline_cycles[0] +
                   t_pipeline_array[1] * t_pipeline_cycles[1] +
                   t_pipeline_array[2] * t_pipeline_cycles[2] +
                   t_pipeline_array[3] * t_pipeline_cycles[3])
        power_avg = (power1 * t_pipeline_array[0] * t_pipeline_cycles[0] * activity[0] +
                     power2 * t_pipeline_array[1] * t_pipeline_cycles[1] * activity[1] +
                     power3 * t_pipeline_array[2] * t_pipeline_cycles[2] * activity[2] +
                     power4 * t_pipeline_array[3] * t_pipeline_cycles[3] * activity[3]) / t_total
        ops_avg = total_ops / t_total
        ops_w_avg = ops_avg / power_avg
    elif get('Train_dataset_type') == 'CIFAR10':
        if get('If_Power_Scale'):
            power1 = get('Conductance_Scale_Factor') * p1 / sum(class_total)
            power2 = get('Conductance_Scale_Factor') * p2 / sum(class_total)
            power3 = get('Conductance_Scale_Factor') * p3 / sum(class_total)
            power4 = get('Conductance_Scale_Factor') * p4 / sum(class_total)
            power5 = get('Conductance_Scale_Factor') * p5 / sum(class_total)
        else:
            power1 = p1 / sum(class_total)
            power2 = p2 / sum(class_total)
            power3 = p3 / sum(class_total)
            power4 = p4 / sum(class_total)
            power5 = p5 / sum(class_total)
        t_total = (t_pipeline_array[0] * t_pipeline_cycles[0] +
                   t_pipeline_array[1] * t_pipeline_cycles[1] +
                   t_pipeline_array[2] * t_pipeline_cycles[2] +
                   t_pipeline_array[3] * t_pipeline_cycles[3] +
                   t_pipeline_array[4] * t_pipeline_cycles[4])
        power_avg = (power1 * t_pipeline_array[0] * t_pipeline_cycles[0] * activity[0] +
                     power2 * t_pipeline_array[1] * t_pipeline_cycles[1] * activity[1] +
                     power3 * t_pipeline_array[2] * t_pipeline_cycles[2] * activity[2] +
                     power4 * t_pipeline_array[3] * t_pipeline_cycles[3] * activity[3] +
                     power5 * t_pipeline_array[4] * t_pipeline_cycles[4] * activity[4]) / t_total
        ops_avg = total_ops / t_total
        ops_w_avg = ops_avg / power_avg
    else:
        sys.exit(1)
    return total_ops, t_total, power_avg, ops_avg, ops_w_avg
