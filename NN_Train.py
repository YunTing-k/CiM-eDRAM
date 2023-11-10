import copy
import os
import time
import numpy as np
import torch
import torch.nn.utils.prune as prune
from tqdm import tqdm
import torchvision.transforms as transforms
import SoftNet
import Utilities as ut
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字


def train(net):
    """Train the input network"""
    train_device = get('Train_device')  # 指定训练设备
    inference_device = get('Inference_device')  # 指定推理设备
    clipweight_range = get('Train_clipweight_range')  # 训练裁剪范围
    """Transform preparation"""
    if get('Train_dataset_type') == 'MNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(18),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    elif get('Train_dataset_type') == 'FashionMNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 Tensor or PIL
            transforms.RandomRotation(8),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    elif get('Train_dataset_type') == 'notMNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(6),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    elif get('Train_dataset_type') == 'CIFAR10':
        custom_transformX = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 Tensor or PIL
            transforms.RandomRotation(10),  # 随机旋转 Tensor or PIL
            # transforms.ColorJitter(brightness=[0.01, 0.05], contrast=[0.3, 0.6], saturation=0.03, hue=0.01),  # 颜色变换
            transforms.ColorJitter(0.08),  # 颜色变换
            transforms.RandomResizedCrop(size=32, scale=(0.6, 1.0)),  # 图片裁剪 Tensor or PIL
            transforms.Normalize([0, 0, 0], [1, 1, 1])
        ])
        # custom_transformY = transforms.Compose([
        #     transforms.Normalize([0, 0, 0], [1, 1, 1])
        # ])
        custom_transformY = None
    else:
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(18),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
        ut.print_warn('Undefined Train_dataset_type', current_name)
    """Dataloader"""
    train_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=custom_transformX, train=True)
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=custom_transformY, train=False)
    train_loader = SoftNet.dataloader_prepare(train_set, train=True, mode='Train')
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode='Train')
    """output data"""
    loss_array = np.zeros((get('Train_epochs'), int(get('Trainset_num') / get('Train_batch'))))  # 损失函数数据
    accuracy_array = np.zeros((get('Train_epochs'), 1))  # 正确率数据
    """loss function and optimizer"""
    criterion = SoftNet.lossfunction_prepare()  # 损失函数定义
    optimizer = SoftNet.adam_optimizer_prepare(net.parameters())  # optimizer定义 ADAM
    net.initialize()  # 权值初始化
    net.train()  # 训练模式

    ut.print_debug('Network training start at ' + time.asctime(), current_name)
    update_count = get('Train_update_count')
    bestAccuracy = 0
    stop_count = 0
    best_Ori_dic = copy.deepcopy(net.state_dict())
    for epoch in range(get('Train_epochs')):  # 训练网络
        net.train()
        running_loss = 0.0
        process_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Soft Training',
                           leave=True, unit='batch')
        for i, (inputs, labels) in process_bar:
            # 获取输入
            inputs = inputs.to(train_device)
            labels = labels.to(train_device)
            # 梯度置0
            optimizer.zero_grad()
            # 正向传播，反向传播，优化
            outputs = net(inputs)  # 正向传播
            loss = criterion(outputs, labels)  # 计算损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权值
            if get('Train_if_clipweight'):  # 裁剪权重
                for layer_name, net_params in net.named_parameters():
                    net_params.data.clamp(clipweight_range[0], clipweight_range[1])
            loss_array[epoch, i] = loss.item()
            running_loss += loss.item()
            if i % update_count == update_count - 1:  # 每固定批次更新一次数据
                process_bar.set_postfix(loss=running_loss / update_count, epoch=epoch + 1, iteration=i + 1)
                running_loss = 0.0
        process_bar.close()
        if get('Early_Stopping'):
            net.eval()  # 推理模式
            # 分类测试网络识别情况
            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    images, labels = data
                    images = images.to(inference_device)
                    labels = labels.to(inference_device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = correct / total
            accuracy_array[epoch, 0] = accuracy
            ut.print_param('%.4f %%' % (accuracy * 100))
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                best_Ori_dic = copy.deepcopy(net.state_dict())
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= get('Early_Stopping_Count'):
                net.load_state_dict(best_Ori_dic)  # 从保存的最好结果中读取数据
                break
        else:
            process_bar.close()
    """保存损失函数"""
    ut.numpy_to_mat(loss_array, get('Train_dataset_type') + 'TrainLoss')
    """保存正确率"""
    ut.numpy_to_mat(accuracy_array, get('Train_dataset_type') + 'TrainAcc')
    ut.print_debug('Training finished at ' + time.asctime(), current_name)
    return net, loss_array, accuracy_array


def prun_train(net):
    """Prune the input network and retrain"""
    train_device = get('Train_device')  # 指定训练设备
    inference_device = get('Inference_device')  # 指定推理设备
    """Transform"""
    if get('Train_dataset_type') == 'MNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(18),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    elif get('Train_dataset_type') == 'FashionMNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 Tensor or PIL
            transforms.RandomRotation(8),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    elif get('Train_dataset_type') == 'notMNIST':
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(6),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
    else:
        custom_transformX = transforms.Compose([
            transforms.RandomRotation(18),  # 随机旋转 Tensor or PIL
            transforms.ColorJitter(0.3),  # 颜色变换 Tensor or PIL [brightness,contrast,saturation,hue]
            transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))  # 图片裁剪 Tensor or PIL
        ])
        ut.print_warn('Undefined Train_dataset_type', current_name)
    """Dataloader"""
    train_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=custom_transformX, train=True)
    test_set = SoftNet.dataset_prepare(dataset_path=get('Train_dataset_path'), transform=None, train=False)
    train_loader = SoftNet.dataloader_prepare(train_set, train=True, mode='Prune')
    test_loader = SoftNet.dataloader_prepare(test_set, train=False, mode='Prune')
    """Preparation"""
    loss_array = np.zeros((get('P_epochs'), int(get('Trainset_num') / get('P_batch'))))
    accuracy_array = np.zeros((get('P_epochs'), 1))  # 正确率数据
    prune_span = get('Prune_iteration_span')  # 剪枝迭代间隔
    prune_step = get('Prune_epoch_step')  # 剪枝步进周期
    p_ratio = get('Prune_ratio')  # 剪枝比例
    p_early_stop_count = get('Prune_Early_Stopping_Count')  # 剪枝早停周期
    criterion = SoftNet.lossfunction_prepare()  # 损失函数定义
    optimizer = SoftNet.P_adam_optimizer_prepare(net.parameters())  # Retrain optimizer定义 ADAM
    ut.print_debug('Network prune with retraining start at ' + time.asctime(), current_name)
    update_count = get('Train_update_count')
    bestAccuracy = 0
    best_Prune_dic = copy.deepcopy(net.state_dict())
    stop_count = 0
    """Sparsity"""
    SoftNet.get_sparsity(net.parameters())
    net.train()  # 训练模式
    for epoch in range(get('P_epochs')):  # 网络再训练
        net.train()  # 训练模式
        running_loss = 0.0
        process_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                           desc='Soft Prune Retraining', leave=True, unit='batch')
        for i, (inputs, labels) in process_bar:
            inputs = inputs.to(train_device)
            labels = labels.to(train_device)
            optimizer.zero_grad()
            outputs = net(inputs)  # 正向传播
            loss = criterion(outputs, labels)  # 计算损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权值
            """Pruning"""
            if ((i + 1) % prune_span == 0) and (epoch < prune_step):  # increasing pruning ratio
                for name, module in net.named_modules():
                    # prune connections in all 2D-conv layers
                    if isinstance(module, torch.nn.Conv2d):
                        if p_ratio[0] != 0:
                            prune.l1_unstructured(module, name='weight',
                                                  amount=p_ratio[0] * ((epoch + 1) / prune_step))
                            prune.remove(module, 'weight')
                    # prune connections in all linear layers
                    elif isinstance(module, torch.nn.Linear):
                        if p_ratio[1] != 0:
                            prune.l1_unstructured(module, name='weight',
                                                  amount=p_ratio[1] * ((epoch + 1) / prune_step))
                            prune.remove(module, 'weight')
            elif ((i + 1) % prune_span == 0) and (epoch >= prune_step):  # fixed pruning ratio
                for name, module in net.named_modules():
                    # prune connections in all 2D-conv layers
                    if isinstance(module, torch.nn.Conv2d):
                        if p_ratio[0] != 0:
                            prune.l1_unstructured(module, name='weight', amount=p_ratio[0])
                            prune.remove(module, 'weight')
                    # prune connections in all linear layers
                    elif isinstance(module, torch.nn.Linear):
                        if p_ratio[1] != 0:
                            prune.l1_unstructured(module, name='weight', amount=p_ratio[1])
                            prune.remove(module, 'weight')
            loss_array[epoch, i] = loss.item()
            running_loss += loss.item()
            if i % update_count == update_count - 1:  # 每固定批次更新一次数据
                process_bar.set_postfix(loss=running_loss / update_count, epoch=epoch + 1, iteration=i + 1)
                running_loss = 0.0
        process_bar.close()
        if get('Prune_Early_Stopping') and epoch >= (prune_step - 1):  # 等待剪枝比例步进结束后再考虑早停
            net.eval()  # 推理模式
            # 分类测试网络识别情况
            correct = 0
            total = 0
            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    images, labels = data
                    images = images.to(inference_device)
                    labels = labels.to(inference_device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            Accuracy = correct / total
            accuracy_array[epoch] = Accuracy
            ut.print_param('%.4f %%' % (Accuracy * 100))
            if Accuracy > bestAccuracy:
                bestAccuracy = Accuracy
                best_Prune_dic = copy.deepcopy(net.state_dict())
                stop_count = 0
            else:
                stop_count += 1
            if stop_count >= p_early_stop_count:
                net.load_state_dict(best_Prune_dic)  # 从保存的最好结果中读取数据
                break
        else:
            process_bar.close()
    ut.print_debug('Training finished at ' + time.asctime(), current_name)
    """保存损失函数"""
    ut.numpy_to_mat(loss_array, get('Train_dataset_type') + 'PruneLoss')
    """保存正确率"""
    ut.numpy_to_mat(accuracy_array, get('Train_dataset_type') + 'PruneAcc')
    return net, loss_array, accuracy_array
