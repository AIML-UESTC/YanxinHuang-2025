# **项目说明**
## Brain Tumor Segmentation
### **BSG-Seg**
#### **BSG-Seg是一个针对Brain stem Glioma的分割案例，数据来源于私有数据集，数据并不规范因此只是一个参考实例，可以用于U-net网络入门。参考以下论文**
>https://doi.org/10.1016/j.mri.2024.07.009
### **Missing-Seg**
#### **Missing-Seg是一个针对脑干胶质瘤的较为全面的研究，后续有诸多可以继续探讨的方向。虽然最终落脚于缺失模态的分割，但是在研究过程中涉及的算法包括图像融合，以及全模态分割等等。代码框架借鉴于**
>https://github.com/HiLab-git/SSL4MIS
#### **同时针对Brats这个面向脑肿瘤的经典数据集，monai这个python库里面包含了许多简易的函数帮助初学者快速上手，建议学习。**
#### **更多详细说明请移步到各个子文件下查看。**
## 3D Cardiac Reconstruction
### **MICCAI-CMR** 
#### **MICCAI-CMR是STACOM依托MICCAI举办的针对3D多模态心脏快速重建算法的一个比赛，是医学图像领域的一个难问题，与传统重建(是指从2D的RGB图像中恢复出3D信息，常用的有神经辐射场等算法)不同，这里的重建算法主要是指基于欠采样(有不同速率的掩码进行采样之后)的心脏频域信息，来恢复原有的全采样的心脏信息(其实属于压缩感知的图像问题)。涉及对高维频域信息的处理，对多模态信息的处理，对算力有较高的要求。代码框架借鉴于**
>https://github.com/hellopipu/PromptMR
#### **此类任务可以产生的重要创新在于如何使用神经网络来完成对欠采样的恢复，需要有较高的数学基础。
>https://ieeexplore.ieee.org/abstract/document/7394156
>https://ieeexplore.ieee.org/abstract/document/7272048(上海市科技进步一等奖)
# 个人主页：
>https://huangyanxin-china.github.io/User-Page/
