## CiM eDRAM项目简介
这是一个基于器件性能出发的仿真器/评估工具，根据器件性能（包括variation，noise）完成从神经网络训练、量化、硬件部署（简单模拟）到阵列性能指标评估。其中存算一体单元主要是eDRAM单元，或电导作为权重存储/计算值的器件都可支持。更详细的内容请参见[论文：](https://)。

This is a computing in memory simulator/evaluator tools. Based on the device's performance/property(including device variation and noise), from CNN training, NN quantization, NN Hardware implementation(rough simulation) and perofrmance evaluation, a complete flow is realized. The basic CiM cell is mainly eDRAM cell or conductance device or other device catagorey that use conductance to store the weight. For more details please see the [corresponding paper: ](https://).

## eDRAM宏单元电路架构
本仿真/评估工具所涉及的具体宏单元架构，采用电流域的存算一体计算架构。
![CiM eDRAM Macro Architecture](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img2.png?raw=true)

## 系统-单元的协同优化与仿真流程
单元系统协同优化(Cell-System Co-optimization Flow)部分在本repo中**没有涉及**，本repo只包含后半部分宏电路仿真(Macro Simulation)。
![CiM eDRAM Simulation Flow](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img1.png?raw=true)

## 目标神经网络
本仿真单元的硬件映射的目标神经网络为基于CIFAR-10与MNSIT数据集的自定义卷积神经网络，下图为数据集为CIFAR-10的自定义卷积神经网络的结构。
![CIFAR-10 CNN](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img3.png?raw=true)

## 卷积算子与全连接算子的映射方式
基于Crossbar结构的并行向量·矩阵乘法的运算，对卷积算子进行稀疏化的映射：
![Operator Mapping](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img6.png?raw=true)

## 第一层卷积层的映射
将第一层卷积层实现用12×10的CiM Tile组合的子阵列映射，每一次计算就是对一个小CiM Tile的激励与输出采样，以避免所有阵列全部打开带来的高功耗以及过高的全局电流密度，同时利用较小的Tile进行计算以避免可能的寄生参数影响。同时为了加快吞吐率，将卷积算子空间上展开8行，并且并行展开8次(对应了第一层卷积算子的64个卷积张量)。
![Conv-1](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img4.png?raw=true)

## 宏单元计算调度
采用逐层调用的计算方式，对于没被调用的层对应的硬件，关闭以减少功耗。
![Schedule](https://github.com/YunTing-k/CiM-eDRAM/blob/master/img/img5.png?raw=true)

## 源代码说明
- **主程序**：DTCO
- **全局参数管理**：GlobalParametersManager
- **工具类**：Utilities
- **硬件参数/函数定义**：Device
- **软件参数/函数定义**：SoftInferenceParameters, SoftTrainParameters
- **软件网络构建/函数定义**：SoftNet
- **硬件映射网络构建/函数定义**：HardNet
- **模型部署参数/函数定义**：ModelDeployment
- **基于SoftTrainParameters给定参数进行网络训练**：NN_Train
- **基于器件数据输入为网络硬件量化**：NN_Quantization
- **基于器件数据输入为网络添加估计噪声**：NN_AddNoise
- **进行软件网络推理/量化后网络推理/硬件部署后的推理/功耗估计推理**：NN_Inference
- **考虑噪声/量化等推理准确度连续测试**：AccuracyTestPipeline
- **考虑器件的Retention Time的推理准确度连续测试**：RetentionTestPipeline

## 项目构建
- Pycharm工程
- 基于Python 3.9.7
- pytorch 1.10, CUDA 11.3, cuDNN8.0
```
- blas=1.0=mkl
- brotli=1.0.9
- ca-certificates=2021.10.26
- certifi=2021.10.8
- cudatoolkit=11.3.1
- cycler=0.11.0
- fonttools=4.25.0
- freetype=2.10.4
- icu=58.2
- intel-openmp=2021.4.0
- jpeg=9d
- kiwisolver=1.3.2
- libpng=1.6.37
- libtiff=4.2.0
- libuv=1.40.0
- libwebp=1.2.0
- lz4-c=1.9.3
- matplotlib=3.5.0
- matplotlib-base=3.5.0
- mkl=2021.4.0
- mkl-service=2.4.0
- mkl_fft=1.3.1
- mkl_random=1.2.2
- munkres=1.1.4
- numpy=1.21.5
- numpy-base=1.21.5
- olefile=0.46
- openssl=1.1.1m
- packaging=21.3
- pillow=8.4.0
- pip=21.2.4
- pyparsing=3.0.4
- pyqt=5.9.2
- python=3.9.7
- python-dateutil=2.8.2
- pytorch=1.10.2=py3.9_cuda11.3_cudnn8_0
- pytorch-mutex=1.0=cuda
- qt=5.9.7
- setuptools=58.0.4
- sip=4.19.13
- six=1.16.0
- sqlite=3.37.2
- tk=8.6.11
- torchaudio=0.10.2
- torchvision=0.11.3
- tornado=6.1
- tqdm=4.62.3
- typing_extensions=3.10.0.2
- tzdata=2021e
- vc=14.2
- vs2015_runtime
- wheel=0.37.1
- wincertstore=0.2
- xz=5.2.5
- zlib=1.2.11
- zstd=1.4.9
- scipy==1.8.0
- torchsummary==1.5.1
```