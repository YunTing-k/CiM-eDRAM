"""
定义软网络的结构
定义相关数据集准备器 优化器的定义
"""
import os
import Utilities as ut
import GlobalParametersManager as gpm
import numpy as np
import scipy.io as io
import torch
import math
import torch.optim as optim
import torch.nn.functional as f
from torch import nn
from torch.utils.data import Dataset, TensorDataset
import torchvision.transforms as transforms

current_name = os.path.basename(__file__)  # 当前模块名字


class MNIST_Type1_Template(nn.Module):
    """适合于MNIST数据集的网络模型\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(MNIST_Type1_Template, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 1176)

        x = self.relu(self.fc1(x))
        x = self.dropout50(x)
        x = self.fc2(x)
        return x

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))
                # nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
                # m.bias.data.zero_()
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, 0, 0.01)


class MNIST_Type1_TemplateU(nn.Module):
    """适合于MNIST数据集的网络模型 剪枝专用\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(MNIST_Type1_TemplateU, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 1176)

        x = self.relu(self.fc1(x))
        x = self.dropout30(x)
        x = self.fc2(x)
        return x


class FashionMNIST_Type1_Template(nn.Module):
    """适合于FashionMNIST数据集的网络模型\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(FashionMNIST_Type1_Template, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2304, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout10(x)
        x = self.relu(self.conv2(x))
        x = self.dropout10(x)
        x = x.view(-1, 2304)

        x = self.relu(self.fc1(x))
        x = self.dropout50(x)
        x = self.fc2(x)
        return x

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))
                # nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
                # m.bias.data.zero_()
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, 0, 0.01)


class FashionMNIST_Type1_TemplateU(nn.Module):
    """适合于FashionMNIST数据集的网络模型 专用于剪枝\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(FashionMNIST_Type1_TemplateU, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2304, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout05(x)
        x = self.relu(self.conv2(x))
        x = self.dropout05(x)
        x = x.view(-1, 2304)

        x = self.relu(self.fc1(x))
        x = self.dropout30(x)
        x = self.fc2(x)
        return x


class notMNIST_Type1_Template(nn.Module):
    """适合于notMNIST数据集的网络模型\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(notMNIST_Type1_Template, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2352, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 2352)

        x = self.relu(self.fc1(x))
        x = self.dropout50(x)
        x = self.fc2(x)
        return x

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))
                # nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
                # m.bias.data.zero_()
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, 0, 0.01)


class notMNIST_Type1_TemplateU(nn.Module):
    """适合于notMNIST数据集的网络模型 专用于剪枝\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(notMNIST_Type1_TemplateU, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2352, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 2352)

        x = self.relu(self.fc1(x))
        x = self.dropout30(x)
        x = self.fc2(x)
        return x


class CIFAR10_Type1_Template(nn.Module):
    """适合于CIFAR10数据集的网络模型\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(CIFAR10_Type1_Template, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=3136, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.fc3 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 3136)

        x = self.relu(self.fc1(x))
        x = self.dropout40(x)
        x = self.relu(self.fc2(x))
        x = self.dropout30(x)
        x = self.fc3(x)
        return x

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data, std=np.sqrt(1 / self.neural_num))
                # nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_out', nonlinearity='relu')
                # m.bias.data.zero_()
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, 0, 0.01)


class CIFAR10_Type1_TemplateU(nn.Module):
    """适合于CIFAR10数据集的网络模型 专用于剪枝\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self):
        super(CIFAR10_Type1_TemplateU, self).__init__()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=3136, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.fc3 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.dropout50 = nn.Dropout(0.50)
        self.dropout45 = nn.Dropout(0.45)
        self.dropout40 = nn.Dropout(0.40)
        self.dropout35 = nn.Dropout(0.35)
        self.dropout30 = nn.Dropout(0.30)
        self.dropout25 = nn.Dropout(0.25)
        self.dropout20 = nn.Dropout(0.20)
        self.dropout15 = nn.Dropout(0.15)
        self.dropout10 = nn.Dropout(0.10)
        self.dropout05 = nn.Dropout(0.05)
        ut.print_info('Software network constructed', current_name)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 3136)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomTrainDatasets(Dataset):
    """自定义训练集"""

    def __init__(self, dataset_path, transform=None, target_transform=None):
        # 读取数据
        data = io.loadmat(dataset_path)
        train_images = data['train_images']
        train_labels = data['train_labels']
        gpm.set_param('Trainset_num', train_labels.shape[0])
        # 转换数据为tensor类型
        train_images = torch.from_numpy(train_images).float() / 255.0  # 转化为0~1的浮点数 以便于后续的transform的使用
        train_labels = torch.from_numpy(train_labels)
        train_labels = torch.squeeze(train_labels).long()
        self.images = train_images
        self.labels = train_labels
        self.transform = transform
        self.target_transform = target_transform
        self.len = train_labels.shape[0]

    def __getitem__(self, index):
        img = self.images[index, :, :, :]  # N C H W
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.len


class CustomTestDatasets(Dataset):
    """自定义测试集"""

    def __init__(self, dataset_path, transform=None, target_transform=None):
        # 读取数据
        data = io.loadmat(dataset_path)
        test_images = data['test_images']
        test_labels = data['test_labels']
        gpm.set_param('Testset_num', test_labels.shape[0])
        # 转换数据为tensor类型
        test_images = torch.from_numpy(test_images).float() / 255.0  # 转化为0~1的浮点数 以便于后续的transform的使用
        test_labels = torch.from_numpy(test_labels)
        test_labels = torch.squeeze(test_labels).long()
        self.images = test_images
        self.labels = test_labels
        self.transform = transform
        self.target_transform = target_transform
        self.len = test_labels.shape[0]

    def __getitem__(self, index):
        img = self.images[index, :, :, :]  # N C H W
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return self.len


def dataset_prepare(dataset_path, transform, train):
    """进行数据集的准备 输入数据集路径 transform 是否为训练集 返回数据集"""
    if train:
        train_set = CustomTrainDatasets(dataset_path=dataset_path, transform=transform)
        ut.print_info('Trainset prepared', current_name)
        return train_set
    else:
        test_set = CustomTestDatasets(dataset_path=dataset_path, transform=transform)
        ut.print_info('Testset prepared', current_name)
        return test_set


def dataloader_prepare(dataset, train, mode):
    """进行Data loader的准备"""
    if mode == 'Train':
        if train:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('Train_batch'),
                                                      shuffle=True, num_workers=4)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('Inference_batch'),
                                                      shuffle=True, num_workers=4)
        ut.print_info('Train data loader prepared', current_name)
        return data_loader
    elif mode == 'Prune':
        if train:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('P_batch'),
                                                      shuffle=False, num_workers=0)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('Inference_batch'),
                                                      shuffle=False, num_workers=0)
        ut.print_info('Prune data loader prepared', current_name)
        return data_loader
    elif mode == 'Quant':
        if train:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('Q_batch'),
                                                      shuffle=False, num_workers=0)
        else:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=gpm.get_param('Inference_batch'),
                                                      shuffle=False, num_workers=0)
        ut.print_info('Quant data loader prepared', current_name)
        return data_loader


def optimizer_prepare(net_parameters):
    """定义训练的optimizer SGD方式"""
    optimizer = optim.SGD(net_parameters,
                          lr=gpm.get_param('Train_lr'),
                          momentum=gpm.get_param('Train_momentum'),
                          dampening=gpm.get_param('Train_dampening'),
                          weight_decay=gpm.get_param('Train_weight_decay'),
                          nesterov=gpm.get_param('Train_nesterov'))
    ut.print_info('Optimizer prepared', current_name)
    return optimizer


def adam_optimizer_prepare(net_parameters):
    """定义训练的optimizer - adam"""
    optimizer = optim.Adam(net_parameters,
                           lr=gpm.get_param('Train_lr'),
                           weight_decay=gpm.get_param('Train_weight_decay'))
    ut.print_info('Optimizer prepared', current_name)
    return optimizer


def P_optimizer_prepare(net_parameters):
    """定义剪枝的optimizer"""
    optimizer = optim.SGD(net_parameters,
                          lr=gpm.get_param('P_lr'),
                          momentum=gpm.get_param('P_momentum'),
                          dampening=gpm.get_param('P_dampening'),
                          weight_decay=gpm.get_param('P_weight_decay'),
                          nesterov=gpm.get_param('P_nesterov'))
    ut.print_info('Prune Retrain Optimizer prepared', current_name)
    return optimizer


def P_adam_optimizer_prepare(net_parameters):
    """定义训练的optimizer - adam"""
    optimizer = optim.Adam(net_parameters,
                           lr=gpm.get_param('P_lr'),
                           weight_decay=gpm.get_param('P_weight_decay'))
    ut.print_info('Prune Retrain Optimizer prepared', current_name)
    return optimizer


def lossfunction_prepare():
    """定义损失函数"""
    if gpm.get_param('Train_loss_type') == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif gpm.get_param('Train_loss_type') == 'L1Loss':
        criterion = nn.L1Loss()
    elif gpm.get_param('Train_loss_type') == 'MSELoss':
        criterion = nn.MSELoss()
    elif gpm.get_param('Train_loss_type') == 'NLLLoss':
        criterion = nn.NLLLoss()
    elif gpm.get_param('Train_loss_type') == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    else:
        criterion = nn.CrossEntropyLoss()
        ut.print_warn('Unknown target loss function,loss function is allocated as CrossEntropyLoss', current_name)
    ut.print_info('Loss function prepared', current_name)
    return criterion


def get_sparsity(parameters):
    """计算神经网络的稀疏度"""
    layer_num = 0  # 层数
    all_zero_num = 0  # 所有非0数目
    all_element_num = 0  # 所有元素数目
    for net_params in parameters:
        layer_num += 1
        zero_num = torch.sum(net_params.data == 0)
        all_zero_num += zero_num
        element_num = net_params.nelement()
        all_element_num += element_num
        sparsity = 100 * float(zero_num) / float(element_num)
        ut.print_param('Sparsity of pruned module-%d is %.3f %%' % (layer_num, sparsity))
    ut.print_param(
        'Sparsity of overall pruned model is %.3f %% with %d zero element and %d total element' %
        (100 * float(all_zero_num) / float(all_element_num), all_zero_num, all_element_num))
