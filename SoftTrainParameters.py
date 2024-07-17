"""定义了软件训练网络的参数 包含训练 剪枝等相关参数"""
import os
import Utilities as ut
import GlobalParametersManager as gpm
import torch

current_name = os.path.basename(__file__)  # 当前模块名字


def set_default_param():
    """设置默认参数"""
    gpm.set_param('SoftNet_class_num', 10)  # 识别任务数目
    gpm.set_param('Train_if_load_model', True)  # 是否从现有文件中加载模型 False~不加载，正常训练 True~加载，不会训练
    gpm.set_param('Inference_skip', True)  # 是否跳过首次推理
    gpm.set_param('TrainSet_Inference', False)  # 是否进行训练集的推理验证

    # gpm.set_param('Train_dataset_type', 'MNIST')  # 数据集名称
    # gpm.set_param('Train_dataset_type', 'FashionMNIST')  # 数据集名称 (截断比例要一起改)
    # gpm.set_param('Train_dataset_type', 'notMNIST')  # 数据集名称
    gpm.set_param('Train_dataset_type', 'CIFAR10')  # 数据集名称

    gpm.set_param('Model_name', 'Custom_' + gpm.get_param('Train_dataset_type'))  # 模型名称
    gpm.set_param('Train_dataset_path', './Data/' + gpm.get_param('Train_dataset_type') + '.mat')  # 数据集路径
    gpm.set_param('Early_Stopping', True)  # 是否早停
    gpm.set_param('Early_Stopping_Count', 20)  # 早停界限
    # 训练批次和次数
    gpm.set_param('Train_epochs', 600)  # 训练次数
    """
    MNIST 50
    FashionMNIST 50
    notMNIST 100
    """
    gpm.set_param('Train_batch', 50)  # batch大小
    # optimizer参数
    gpm.set_param('Train_lr', 0.0005)  # 学习率
    gpm.set_param('Train_lr_scheduler', False)  # 训练学习率scheduler设置 True:SGD+Scheduler False:Adam
    gpm.set_param('Train_momentum', 0.90)  # 惯性系数
    gpm.set_param('Train_dampening', 0)  # 动量抑制因子
    gpm.set_param('Train_weight_decay', 0)  # L2正则惩罚
    gpm.set_param('Train_nesterov', True)  # 是否使用nesterov动量
    # loos函数类别
    gpm.set_param('Train_loss_type', 'CrossEntropyLoss')  # 默认为交叉熵函数
    # 训练使用device
    gpm.set_param('Train_device', 'cuda:0')  # 默认使用GPU训练
    # gpm.set_param('Train_device', 'cpu')  # 默认使用CPU推理
    # 权重裁剪
    gpm.set_param('Train_if_clipweight', False)  # 是否裁剪权重
    gpm.set_param('Train_clipweight_range', (-1, 1))  # 裁剪权重范围
    # 剪枝相关参数
    gpm.set_param('If_prune', False)  # 是否剪枝，如果是，则进行剪枝，如果不是则直接跳过
    # gpm.set_param('If_prune', False)  # 是否剪枝，如果是，则进行剪枝，如果不是则直接跳过
    gpm.set_param('Prune_retrain', False)  # 是否进行剪枝重新训练 是则重新训练 否则读取文件
    gpm.set_param('Prune_Early_Stopping', True)  # 是否早停-剪枝
    gpm.set_param('Prune_Early_Stopping_Count', 20)  # 早停界限-剪枝
    gpm.set_param('Prune_inference_skip', True)  # 是否跳过剪枝的后的推理
    gpm.set_param('Prune_iteration_span', 100)  # 剪枝迭代周期 每一个epoch的循环内 进行以Prune_iteration_span为周期的剪枝操作
    gpm.set_param('Prune_epoch_step', 20)  # 剪枝迭代周期 以  1 / Prune_step 为步长步进剪枝比例至设定值
    # 剪枝比例于 ModelCompress模块里调整
    gpm.set_param('P_epochs', 600)  # 再训练次数
    """
        MNIST 50
        FashionMNIST 50
        notMNIST 100
    """
    gpm.set_param('P_batch', 50)  # batch大小
    gpm.set_param('P_lr', 0.0005)  # 学习率
    gpm.set_param('P_lr_scheduler', False)  # 剪枝学习率scheduler设置 True:SGD+Scheduler False:Adam
    gpm.set_param('P_momentum', 0.90)  # 惯性系数
    gpm.set_param('P_dampening', 0)  # 动量抑制因子
    gpm.set_param('P_weight_decay', 0)  # L2正则惩罚
    gpm.set_param('P_nesterov', True)  # 是否使用nesterov动量
    gpm.set_param('Q_skip', False)  # 是否跳过量化 跳过量化则一并跳过噪声推理
    gpm.set_param('Q_Read', False)  # 是否从文件中读取量化后的权重以及各个参数
    gpm.set_param('Q_inference_skip', False)  # 是否跳过量化推理 在Q_Read = True有用
    gpm.set_param('Q_batch', 50)  # batch大小
    # 训练设备
    if gpm.get_param('Train_device') == 'cuda:0':
        if not torch.cuda.is_available():  # 无法调用CUDA
            gpm.set_param('Train_device', 'cpu')
            ut.print_warn('Cuda device is unavailable', current_name)
    ut.print_info('Current device is:' + gpm.get_param('Train_device'), current_name)
    # 显示调整
    gpm.set_param('Train_update_count', 40)  # 训练每批次输出数据
    gpm.set_param('If_display_distribution', False)  # 是否展示分布

    ut.print_info('Softnet train parameters added', current_name)
