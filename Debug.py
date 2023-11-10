"""测试程序"""
import os
import Utilities as ut
import numpy as np
import torch
from tqdm import tqdm
import time
from torch import nn
from torchvision import transforms
import winsound

current_name = os.path.basename(__file__)  # 当前模块名字
if __name__ == '__main__':
    a = np.array([[1,2,3],[3,2,1]])
    b = np.array([[10,1000],[100,100],[1000,10]])
    print(np.dot(a,b))
    print(np.mat(a) * np.mat(b))
    print(a @ b)
    print(np.matmul(a,b))
