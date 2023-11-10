"""
定义硬件网络
"""
import os
import copy
import sys

import Device
import numpy as np
import Utilities as ut
import torch
from torch import nn
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字


class MNIST_Type1_Template_Synapse(nn.Module):
    """适合于MNIST数据集的网络模型 在展开的映射算法以及考虑突触特性的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1w, conv2w, fc1w, fc2w, param, time):
        super(MNIST_Type1_Template_Synapse, self).__init__()

        self.relu = nn.ReLU()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.conv1.weight.data = conv1w  # conv1权重
        self.conv2.weight.data = conv2w  # conv2权重
        self.fc1.weight.data = fc1w  # fc1权重
        self.fc2.weight.data = fc2w  # fc2权重
        self.beta = param * time  # 时间和突触增益的乘积
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        """考虑突触特性的运算"""
        out1 = self.relu(self.conv1(x) + 4 * self.beta * self.a_pool(torch.mul(x, x)))
        """其余的运算"""
        out2 = self.relu(self.conv2(out1))
        out2 = out2.view(-1, 1176)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class MNIST_Type1_Template_Unroll(nn.Module):
    """适合于MNIST数据集的网络模型 在展开的映射算法下的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1, conv2, fc1, fc2):  # 输入卷积核参数 注意一些需要权值共享 一些不需要权值共享
        super(MNIST_Type1_Template_Unroll, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.fc1 = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.conv1w = conv1  # conv1权重(展开)
        self.conv2w = conv2  # conv2权重(展开)
        self.fc1.weight.data = fc1  # fc1权重(不展开)
        self.fc2.weight.data = fc2  # fc2权重(不展开)
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        batch = x.shape[0]  # 取得mini batch的大小
        """第一层卷积展开"""
        out1 = torch.zeros(batch, 3, 14, 14, dtype=torch.float).to(get('Train_device'))  # N C H W N*1*28*28 → N*3*14*14
        for channel in range(3):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(14):
                for width in range(14):
                    self.conv1.weight.data = self.conv1w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv1(x))  # 1 C H W
                    out1[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """第二层卷积展开"""
        out2 = torch.zeros(batch, 24, 7, 7, dtype=torch.float).to(get('Train_device'))  # N C H W N*3*14*14 → N*24*7*7
        for channel in range(24):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(7):
                for width in range(7):
                    self.conv2.weight.data = self.conv2w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv2(out1))  # 1 C H W
                    out2[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """不展开的运算"""
        out2 = out2.view(-1, 1176)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class MNIST_Type1_Template_Power(nn.Module):
    """适合于MNIST数据集的网络模型 估计推理功耗\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self,
                 conv1p, conv2p, fc1p, fc2p,  # 功率域权值
                 conv1i, conv2i, fc1i, fc2i,  # 电流域权值
                 input_max):  # 输入转换系数 灰度值转电压
        super(MNIST_Type1_Template_Power, self).__init__()
        self.relu = nn.ReLU()

        self.conv1p = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2p = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1p = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2p = nn.Linear(in_features=512, out_features=10, bias=False)

        self.conv1i = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2i = nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1i = nn.Linear(in_features=1176, out_features=512, bias=False)
        self.fc2i = nn.Linear(in_features=512, out_features=10, bias=False)

        self.conv1p.weight.data = torch.unsqueeze(conv1p, 1)  # conv1功率域权重
        self.conv2p.weight.data = conv2p  # conv2功率域权重
        self.fc1p.weight.data = fc1p  # fc1功率域权重
        self.fc2p.weight.data = fc2p  # fc2功率域权重

        self.conv1i.weight.data = torch.unsqueeze(conv1i, 1)  # conv1电流域权重
        self.conv2i.weight.data = conv2i  # conv2电流域权重
        self.fc1i.weight.data = fc1i  # fc1电流域权重
        self.fc2i.weight.data = fc2i  # fc2电流域权重
        self.input_max = input_max
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        """定义功率估计过程"""
        """第一层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        p = self.conv1p(torch.pow(x * self.input_max, 2))  # 功率域结果 W (输入映射到0~Max_vin) x原始输入介于0~1
        i = self.relu(self.conv1i(x * self.input_max))  # 电流域结果 A (输入映射到0~Max_vin)
        power1 = torch.sum(p)  # 第一层功耗 W
        """第二层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        vv = torch.pow(v, 2)
        p = self.conv2p(vv)  # 功率域结果 W
        i = self.relu(self.conv2i(v))  # 电流域结果 A
        power2 = torch.sum(p)  # 第二层功耗 W
        """第三层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        v = v.view(-1, 1176)
        vv = torch.pow(v, 2)
        p = self.fc1p(vv)  # 功率域结果 W
        i = self.relu(self.fc1i(v))  # 电流域结果 A
        power3 = torch.sum(p)  # 第三层功耗 W
        """第四层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        vv = torch.pow(v, 2)
        p = self.fc2p(vv)  # 功率域结果 W
        out = self.fc2i(v)
        power4 = torch.sum(p)  # 第四层功耗 W
        return power1, power2, power3, power4, out


class FashionMNIST_Type1_Template_Synapse(nn.Module):
    """适合于FashionMNIST数据集的网络模型 在展开的映射算法以及考虑突触特性的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1w, conv2w, fc1w, fc2w, param, time):
        super(FashionMNIST_Type1_Template_Synapse, self).__init__()

        self.relu = nn.ReLU()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2304, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.conv1.weight.data = conv1w  # conv1权重
        self.conv2.weight.data = conv2w  # conv2权重
        self.fc1.weight.data = fc1w  # fc1权重
        self.fc2.weight.data = fc2w  # fc2权重
        self.beta = param * time  # 时间和突触增益的乘积
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        """考虑突触特性的运算"""
        out1 = self.relu(self.conv1(x) + 4 * self.beta * self.a_pool(torch.mul(x, x)))
        """其余的运算"""
        out2 = self.relu(self.conv2(out1))
        out2 = out2.view(-1, 2304)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class FashionMNIST_Type1_Template_Unroll(nn.Module):
    """适合于FashionMNIST数据集的网络模型 在展开的映射算法下的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1, conv2, fc1, fc2):  # 输入卷积核参数 注意一些需要权值共享 一些不需要权值共享
        super(FashionMNIST_Type1_Template_Unroll, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(4, 4), stride=(2, 2), bias=False)  # 需要展开
        self.fc1 = nn.Linear(in_features=2304, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.conv1w = conv1  # conv1权重(展开)
        self.conv2w = conv2  # conv2权重(展开)
        self.fc1.weight.data = fc1  # fc1权重(不展开)
        self.fc2.weight.data = fc2  # fc2权重(不展开)
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        batch = x.shape[0]  # 取得mini batch的大小
        """第一层卷积展开"""
        out1 = torch.zeros(batch, 3, 14, 14, dtype=torch.float).to(get('Train_device'))  # N C H W N*1*28*28 → N*3*14*14
        for channel in range(3):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(14):
                for width in range(14):
                    self.conv1.weight.data = self.conv1w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv1(x))  # 1 C H W
                    out1[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """第二层卷积展开"""
        out2 = torch.zeros(batch, 64, 6, 6, dtype=torch.float).to(get('Train_device'))  # N C H W N*3*14*14 → N*64*6*6
        for channel in range(64):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(6):
                for width in range(6):
                    self.conv2.weight.data = self.conv2w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv2(out1))  # 1 C H W
                    out2[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """不展开的运算"""
        out2 = out2.view(-1, 2304)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class notMNIST_Type1_Template_Synapse(nn.Module):
    """适合于notMNIST数据集的网络模型 在展开的映射算法以及考虑突触特性的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1w, conv2w, fc1w, fc2w, param, time):
        super(notMNIST_Type1_Template_Synapse, self).__init__()

        self.relu = nn.ReLU()
        self.a_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.fc1 = nn.Linear(in_features=2352, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.conv1.weight.data = conv1w  # conv1权重
        self.conv2.weight.data = conv2w  # conv2权重
        self.fc1.weight.data = fc1w  # fc1权重
        self.fc2.weight.data = fc2w  # fc2权重
        self.beta = param * time  # 时间和突触增益的乘积
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        """考虑突触特性的运算"""
        out1 = self.relu(self.conv1(x) + 4 * self.beta * self.a_pool(torch.mul(x, x)))
        """其余的运算"""
        out2 = self.relu(self.conv2(out1))
        out2 = out2.view(-1, 2352)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class notMNIST_Type1_Template_Unroll(nn.Module):
    """适合于FashionMNIST数据集的网络模型 在展开的映射算法下的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1, conv2, fc1, fc2):  # 输入卷积核参数 注意一些需要权值共享 一些不需要权值共享
        super(notMNIST_Type1_Template_Unroll, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.fc1 = nn.Linear(in_features=2352, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=False)
        self.conv1w = conv1  # conv1权重(展开)
        self.conv2w = conv2  # conv2权重(展开)
        self.fc1.weight.data = fc1  # fc1权重(不展开)
        self.fc2.weight.data = fc2  # fc2权重(不展开)
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        batch = x.shape[0]  # 取得mini batch的大小
        """第一层卷积展开"""
        out1 = torch.zeros(batch, 3, 14, 14, dtype=torch.float).to(get('Train_device'))  # N C H W N*1*28*28 → N*3*14*14
        for channel in range(3):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(14):
                for width in range(14):
                    self.conv1.weight.data = self.conv1w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv1(x))  # 1 C H W
                    out1[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """第二层卷积展开"""
        out2 = torch.zeros(batch, 48, 7, 7, dtype=torch.float).to(get('Train_device'))  # N C H W N*3*14*14 → N*48*7*7
        for channel in range(48):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(7):
                for width in range(7):
                    self.conv2.weight.data = self.conv2w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv2(out1))  # 1 C H W
                    out2[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """不展开的运算"""
        out2 = out2.view(-1, 2352)
        x = self.relu(self.fc1(out2))
        x = self.fc2(x)
        return x


class CIFAR10_Type1_Template_Unroll(nn.Module):
    """适合于CIFAR10数据集的网络模型 在展开的映射算法下的网络\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self, conv1, conv2, fc1, fc2, fc3):  # 输入卷积核参数 注意一些需要权值共享 一些不需要权值共享
        super(CIFAR10_Type1_Template_Unroll, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(2, 2), stride=(2, 2), bias=False)  # 需要展开
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(4, 4), stride=(2, 2), bias=False)  # 需要展开
        self.fc1 = nn.Linear(in_features=3136, out_features=1024, bias=False)
        self.fc2 = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.fc3 = nn.Linear(in_features=512, out_features=10, bias=False)
        self.conv1w = conv1  # conv1权重(展开)
        self.conv2w = conv2  # conv2权重(展开)
        self.fc1.weight.data = fc1  # fc1权重(不展开)
        self.fc2.weight.data = fc2  # fc2权重(不展开)
        self.fc3.weight.data = fc3  # fc2权重(不展开)
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        batch = x.shape[0]  # 取得mini batch的大小
        """第一层卷积展开"""
        out1 = torch.zeros(batch, 64, 16, 16, dtype=torch.float).to(get('Train_device'))  # N C H W N*3*32*32→N*64*16*16
        for channel in range(64):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(16):
                for width in range(16):
                    self.conv1.weight.data = self.conv1w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv1(x))  # 1 C H W
                    out1[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """第二层卷积展开"""
        out2 = torch.zeros(batch, 64, 7, 7, dtype=torch.float).to(get('Train_device'))  # N C H W N*64*16*16 → N*64*7*7
        for channel in range(64):
            weight_itr = 0  # 该channel计算完毕 计数归零
            for height in range(7):
                for width in range(7):
                    self.conv2.weight.data = self.conv2w[channel, weight_itr].unsqueeze(
                        0)  # 载入权值 载入第channel行 第weight_itr的权值
                    tmp = self.relu(self.conv2(out1))  # 1 C H W
                    out2[:, channel, height, width] = tmp[:, 0, height, width]  # N C H W
                    weight_itr = weight_itr + 1  # 下一个权重
        """不展开的运算"""
        out2 = out2.view(-1, 3136)
        x = self.relu(self.fc1(out2))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR10_Type1_Template_Power(nn.Module):
    """适合于MNIST数据集的网络模型 估计推理功耗\n
    CIS+Template CONV → CIM+FC"""

    def __init__(self,
                 conv1p, conv2p, fc1p, fc2p, fc3p,  # 功率域权值
                 conv1i, conv2i, fc1i, fc2i, fc3i,  # 电流域权值
                 input_max):  # 输入转换系数 灰度值转电压
        super(CIFAR10_Type1_Template_Power, self).__init__()
        self.relu = nn.ReLU()

        self.conv1p = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2p = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1p = nn.Linear(in_features=3136, out_features=1024, bias=False)
        self.fc2p = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.fc3p = nn.Linear(in_features=512, out_features=10, bias=False)

        self.conv1i = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2i = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.fc1i = nn.Linear(in_features=3136, out_features=1024, bias=False)
        self.fc2i = nn.Linear(in_features=1024, out_features=512, bias=False)
        self.fc3i = nn.Linear(in_features=512, out_features=10, bias=False)

        self.conv1p.weight.data = conv1p  # conv1功率域权重
        self.conv2p.weight.data = conv2p  # conv2功率域权重
        self.fc1p.weight.data = fc1p  # fc1功率域权重
        self.fc2p.weight.data = fc2p  # fc2功率域权重
        self.fc3p.weight.data = fc3p  # fc3功率域权重

        self.conv1i.weight.data = conv1i  # conv1电流域权重
        self.conv2i.weight.data = conv2i  # conv2电流域权重
        self.fc1i.weight.data = fc1i  # fc1电流域权重
        self.fc2i.weight.data = fc2i  # fc2电流域权重
        self.fc3i.weight.data = fc3i  # fc3电流域权重
        self.input_max = input_max
        ut.print_info('Hardware network constructed', current_name)

    def forward(self, x):
        """定义功率估计过程"""
        """第一层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        p = self.conv1p(torch.pow(x * self.input_max, 2))  # 功率域结果 W (输入映射到0~Max_vin) x原始输入介于0~1
        i = self.relu(self.conv1i(x * self.input_max))  # 电流域结果 A (输入映射到0~Max_vin)
        power1 = torch.sum(p)  # 第一层功耗 W
        """第二层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        vv = torch.pow(v, 2)
        p = self.conv2p(vv)  # 功率域结果 W
        i = self.relu(self.conv2i(v))  # 电流域结果 A
        power2 = torch.sum(p)  # 第二层功耗 W
        """第三层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        v = v.view(-1, 3136)
        vv = torch.pow(v, 2)
        p = self.fc1p(vv)  # 功率域结果 W
        i = self.relu(self.fc1i(v))  # 电流域结果 A
        power3 = torch.sum(p)  # 第三层功耗 W
        """第四层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        vv = torch.pow(v, 2)
        p = self.fc2p(vv)  # 功率域结果 W
        i = self.relu(self.fc2i(v))  # 电流域结果 A
        power4 = torch.sum(p)  # 第四层功耗 W
        """第五层
        P = (G+ + G-) * (V x V) ; I = (G+ - G-) x V ; V = I * amp
        """
        v = i * (self.input_max / torch.max(i))  # (输出映射到0~Max_vin)
        vv = torch.pow(v, 2)
        p = self.fc3p(vv)  # 功率域结果 W
        out = self.fc3i(v)  # 电流域结果
        power5 = torch.sum(p)  # 第四层功耗 W
        return power1, power2, power3, power4, power5, out


def power_net_prepare(in_net, device_file, mode):
    """Build up the NN for power estimation"""
    """Preparation"""
    if get('Quant_type') == 'Bias':  # CIM 偏置型
        ut.print_error('Bias is not supported for power estimation', current_name)
        sys.exit(1)
    train_device = get('Train_device')  # 指定训练设备
    Discrete_Calibration_order = get('Discrete_Calibration_order')  # 离散器件标定级数
    """Read data"""
    #  量化后的权重对应量化表的下标 软件域
    quantized_tag_lut_sets = np.load('./Parameters/quantized_tag_lut_sets.npy', allow_pickle=True).item()
    #  每一个ANN权重对应的量化表的下标
    quantized_tensor_tag = np.load('./Parameters/quantized_tensor_tag.npy', allow_pickle=True).item()
    #  量化到软件域的scale系数
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
        cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
            device_file + get('Train_dataset_type') + 'Single.mat', Discrete_Calibration_order)  # 得到原始数据 查找表
    else:
        pass
    current_layer = 0
    for net_params in net.parameters():  # 遍历网络层进行量化
        if mode == 'global':
            cim_curve_cluster, cim_status_num, cim_cali_curve, _, _, _ = Device.discrete_calibrate_curve(
                device_file + get('Train_dataset_type') + str(current_layer) + '.mat',
                Discrete_Calibration_order)  # 得到原始数据 查找表
        else:
            pass
        q_index = quantized_tensor_tag[layer_name[current_layer]]
        w_plus_tag = quantized_tag_lut_sets[layer_name[current_layer] + 'plus']
        w_minus_tag = quantized_tag_lut_sets[layer_name[current_layer] + 'minus']
        if current_layer == 0:
            w_plus = np.squeeze(cim_cali_curve[w_plus_tag[q_index]])
            w_minus = np.squeeze(cim_cali_curve[w_minus_tag[q_index]])
            conv1p = torch.from_numpy((w_plus + w_minus))  # (G+ + G-)
            conv1p = conv1p.type(torch.FloatTensor).to(train_device)
            # conv1i = net_params.data / quant_scale_sets[current_layer, 0]  # QuantedWeight = Physics * scale
            conv1i = torch.from_numpy((w_plus - w_minus))  # Effective Physical Conductance
            conv1i = conv1i.type(torch.FloatTensor).to(train_device)
        elif current_layer == 1:
            w_plus = np.squeeze(cim_cali_curve[w_plus_tag[q_index]])
            w_minus = np.squeeze(cim_cali_curve[w_minus_tag[q_index]])
            conv2p = torch.from_numpy((w_plus + w_minus))  # (G+ + G-)
            conv2p = conv2p.type(torch.FloatTensor).to(train_device)
            # conv2i = net_params.data / quant_scale_sets[current_layer, 0]  # QuantedWeight = Physics * scale
            conv2i = torch.from_numpy((w_plus - w_minus))  # Effective Physical Conductance
            conv2i = conv2i.type(torch.FloatTensor).to(train_device)
        elif current_layer >= 2:
            w_plus = np.squeeze(cim_cali_curve[w_plus_tag[q_index]])
            w_minus = np.squeeze(cim_cali_curve[w_minus_tag[q_index]])
            fcp = torch.from_numpy((w_plus + w_minus))  # (G+ + G-)
            fcp = fcp.type(torch.FloatTensor).to(train_device)
            # fci = net_params.data / quant_scale_sets[current_layer, 0]  # QuantedWeight = Physics * scale
            fci = torch.from_numpy((w_plus - w_minus))  # Effective Physical Conductance
            fci = fci.type(torch.FloatTensor).to(train_device)
            if current_layer == 2:
                fc1p = fcp
                fc1i = fci
            elif current_layer == 3:
                fc2p = fcp
                fc2i = fci
            else:
                fc3p = fcp
                fc3i = fci
        current_layer = current_layer + 1
    if get('Train_dataset_type') == 'MNIST':
        power_net = MNIST_Type1_Template_Power(conv1p=conv1p, conv2p=conv2p, fc1p=fc1p, fc2p=fc2p,
                                               conv1i=conv1i, conv2i=conv2i, fc1i=fc1i, fc2i=fc2i,
                                               input_max=get('Max_vin'))
    elif get('Train_dataset_type') == 'CIFAR10':
        power_net = CIFAR10_Type1_Template_Power(conv1p=conv1p, conv2p=conv2p, fc1p=fc1p, fc2p=fc2p, fc3p=fc3p,
                                                 conv1i=conv1i, conv2i=conv2i, fc1i=fc1i, fc2i=fc2i, fc3i=fc3i,
                                                 input_max=get('Max_vin'))
    if mode == 'single':
        ut.print_info('Power-net for single quantization is prepared', current_name)
    elif mode == 'global':
        ut.print_info('Power-net for global quantization is prepared', current_name)
    power_net_ = copy.deepcopy(power_net)
    return power_net_


def capacitance_net(in_net, device_file, mode):
    """Build up the NN for power estimation"""
    """Preparation"""
    if get('Quant_type') == 'Bias':  # CIM 偏置型
        ut.print_error('Bias is not supported for power estimation', current_name)
        sys.exit(1)
    """Read data"""
    #  量化后的权重对应量化表的下标 软件域
    quantized_tag_lut_sets = np.load('./Parameters/quantized_tag_lut_sets.npy', allow_pickle=True).item()
    #  每一个ANN权重对应的量化表的下标
    quantized_tensor_tag = np.load('./Parameters/quantized_tensor_tag.npy', allow_pickle=True).item()
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
        area_lut = ut.mat_to_numpy(device_file + get('Train_dataset_type') + 'Single_Area.mat', 'area')
    else:
        pass
    if get('Train_dataset_type') == 'MNIST':
        total_refresh_time = (19.6 / get('Clock_freq')) +\
                             ((4.9 * 6) / get('Clock_freq')) +\
                             ((51.2 * 11.76 / 4) / get('Clock_freq')) +\
                             ((5.12 / 4) / get('Clock_freq'))
        computing_area = 2 * 10 * [461, 461, 657, 541]
        area_unroll_num = [4 * 10 * 19.6 * 3, 12 * 10 * 4.9 * 6, 1176 * 512, 512 * 10]  # 每一层器件数目
    elif get('Train_dataset_type') == 'CIFAR10':
        total_refresh_time = ((8 * 25.6) / get('Clock_freq')) + \
                             ((4.9 * 2.56 * 64 / 8) / get('Clock_freq')) + \
                             ((102.4 * 31.36 / 8) / get('Clock_freq')) + \
                             ((51.2 * 10.24 / 8) / get('Clock_freq')) +\
                             ((51.2 / 4) / get('Clock_freq'))
        computing_area = 2 * 10 * [1288, 1415, 1128, 1249, 562]
        area_unroll_num = [12 * 10 * 25.6 * 16 * 4, 256 * 64 * 49, 3136 * 1024, 1024 * 512, 512 * 10]  # 每一层器件数目
    current_layer = 0
    total_cap_area = 0
    all_area = 0
    for net_params in net.parameters():  # 遍历网络层进行量化
        if mode == 'global':
            area_lut = ut.mat_to_numpy(device_file + get('Train_dataset_type') + str(current_layer) + '_Area.mat', 'area')
        else:
            pass
        q_index = quantized_tensor_tag[layer_name[current_layer]]
        w_plus_tag = quantized_tag_lut_sets[layer_name[current_layer] + 'plus']
        w_minus_tag = quantized_tag_lut_sets[layer_name[current_layer] + 'minus']
        cap_plus = area_lut[w_plus_tag[q_index]]
        cap_minus = area_lut[w_minus_tag[q_index]]
        cap_area = np.sum(cap_plus + cap_minus) * (area_unroll_num[current_layer] / np.size(cap_plus))
        total_cap_area = total_cap_area + cap_area
        if mode == 'global':
            all_area = all_area + area_unroll_num[current_layer] * computing_area[current_layer] + area_unroll_num[current_layer] * get('Read_Area')
        else:
            all_area = all_area + area_unroll_num[current_layer] * (1500 * 10 * 2) + area_unroll_num[current_layer] * get('Read_Area')
        current_layer = current_layer + 1
    if get('If_Area_Scale'):
        total_cap_area = total_cap_area * (get('Width_Scale_factor') * get('Length_Scale_Factor'))
        all_area = all_area * (get('Width_Scale_factor') * get('Length_Scale_Factor'))
    total_refresh_energy = 0.5 * (total_cap_area * get('Unit_cap') * 1e-12) * (get('Store_voltage') ** 2)
    if mode == 'single':
        ut.print_info('Capacitance estimation for single quantization is prepared', current_name)
    elif mode == 'global':
        ut.print_info('Capacitance estimation for global quantization is prepared', current_name)

    return total_cap_area, total_refresh_energy, total_refresh_time, all_area
