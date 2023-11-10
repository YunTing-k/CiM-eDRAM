import copy
import os
import time
import numpy as np
import torch
import torch.nn.utils.prune as prune
import winsound
from tqdm import tqdm
from torchsummary import summary
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

import GlobalParametersManager as gpm
import PerformanceEvaluation
import SoftInferenceParameters
import SoftTrainParameters
import ModelDeployment
import Device
import SoftNet
import Mask
import Utilities as ut
from GlobalParametersManager import get_param as get

current_name = os.path.basename(__file__)  # 当前模块名字
if __name__ == '__main__':
    a = np.logspace(0, 3, 20, base=10)
    print(a)

