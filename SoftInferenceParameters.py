"""定义了软件训练网络的参数"""
import os
import Utilities as ut
import GlobalParametersManager as gpm
import torch

current_name = os.path.basename(__file__)  # 当前模块名字


def set_default_param():
    """设置默认参数"""
    gpm.set_param('Inference_device', 'cuda:0')  # 默认使用GPU推理
    # gpm.set_param('Inference_device', 'cpu')  # 默认使用CPU推理
    if gpm.get_param('Inference_device') == 'cuda:0':
        if not torch.cuda.is_available():  # 无法调用CUDA
            gpm.set_param('Inference_device', 'cpu')
            ut.print_warn('Cuda device is unavailable', current_name)
    gpm.set_param('Inference_batch', 500)  # 推理batch size大小
    ut.print_info('Softnet inference parameters added', current_name)
