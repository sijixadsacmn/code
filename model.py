# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import torch
import torch.nn as nn
from torch.nn import parameter
import numpy as np
from utils import HistHash
from torch.nn import functional as F
from math import floor,ceil

class RN2DPCA(nn.Module):
    def __init__(self, device, in_channels, kernel_size1, kernel_size2, rff_num_filter1, 
                 rff_num_filter2, pca_num_filter1, pca_num_filter2, hist_blk_size,
                 hist_blk_over, sigma1, sigma2):
        super(RN2DPCA, self).__init__()
        self.ssp = SPP(5)
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.rff_num_filter1 = rff_num_filter1
        self.rff_num_filter2 = rff_num_filter2
        self.pca_num_filter1 = pca_num_filter1
        self.pca_num_filter2 = pca_num_filter2
        self.device =device
        self.hist_hash = HistHash(
            device, pca_num_filter2, hist_blk_size, hist_blk_over)
        #用于去平均值的块
        self.mean_conv1 = nn.Conv2d(
            in_channels=in_channels,#1,1,7*7
            out_channels=in_channels,
            kernel_size=kernel_size1,
            padding=round((kernel_size1-1)/2),
            bias=False
        ).requires_grad_(False)

        nn.init.constant_(self.mean_conv1.weight, 1.0 /
                          (kernel_size1**2 * in_channels))#初始化网络权重（求均值块） 权重是1/49，也就是对每个块求和乘1/49
        #
        self.mean_mul1 = nn.Conv2d(
            in_channels=in_channels,#1,50,1
            out_channels=rff_num_filter1,#D1维度50
            kernel_size=1,#kernel
            bias=True#b
        ).requires_grad_(False)

        self.rff1 = nn.Conv2d(
            in_channels=in_channels,#1,50,7
            out_channels=rff_num_filter1,
            kernel_size=kernel_size1,
            padding=round((kernel_size1-1)/2)
        ).requires_grad_(False)

        self.pca1 = nn.Linear(
            in_features=rff_num_filter1,
            out_features=pca_num_filter1,
            bias=False
        ).requires_grad_(False)

        self.mean_conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size2,
            padding=round((kernel_size2-1)/2),
            bias=False
        ).requires_grad_(False)

        nn.init.constant_(self.mean_conv2.weight, 1.0/(kernel_size2**2))

        self.mean_mul2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=rff_num_filter2,
            kernel_size=1,
            bias=True
        ).requires_grad_(False)

        self.rff2 = nn.Conv2d(
            in_channels=1,
            out_channels=rff_num_filter2,
            kernel_size=kernel_size2,
            padding=round((kernel_size2-1)/2)
        ).requires_grad_(False)

        self.pca2 = nn.Linear(
            in_features=rff_num_filter2,
            out_features=pca_num_filter2,
            bias=False
        ).requires_grad_(False)
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, 
            stride=2
        )

    def forward(self, x, return_loc=None,obj=None):

        batch_size = x.shape[0]
        # only if the size of the image won't change
        dim1, dim2 = x.shape[2], x.shape[3]
        x_mean = self.mean_conv1(x)#按块去均值
        self.mean_mul1.weight = parameter.Parameter(torch.sum(
            torch.sum(self.rff1.weight, dim=3), dim=2).unsqueeze(2).unsqueeze(3))#
        x_mean = self.mean_mul1(x_mean)
        x = self.rff1(x)
        x = x - x_mean  
        x = np.sqrt(2/self.rff_num_filter1) * torch.cos(x)
        if return_loc=='rff1':
            return x
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, dim1*dim2, self.rff_num_filter1)

        x = self.pca1(x)
        x = x.permute(0, 2, 1)#1,8,16384>1,8,16384

        x = x.reshape(batch_size, self.pca_num_filter1, dim1, dim2)#1,8,16384>1,8,128,128
        layer = nn.BatchNorm2d(x.shape[1]).to(device=self.device) 
        x = layer(x)
        x = nn.GELU()(x)
        # a trick based on creating minibatch to convolve with each channel
        x = x.reshape(batch_size*self.pca_num_filter1, 1, dim1, dim2)#8,1,128,128
        #print('x5shape',x.shape)
        x_mean = self.mean_conv2(x)#8,1,128,128
        self.mean_mul2.weight = parameter.Parameter(torch.sum(
            torch.sum(self.rff2.weight, dim=3), dim=2).unsqueeze(2).unsqueeze(3))
        x_mean = self.mean_mul2(x_mean)  # perform w^T.E[x]
        x = self.rff2(x)#8,50,128,128

        x = x - x_mean # mean removal
        x = np.sqrt(2/self.rff_num_filter2) * torch.cos(x)
        if return_loc=='rff2':
            return x
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], dim1*dim2, self.rff_num_filter1)
        x = self.pca2(x)#50>8
        
        x = x.unsqueeze(0)
        layer = nn.BatchNorm2d(x.shape[1]).to(device=self.device) # 传入通道数
        x = layer(x)
        x = nn.GELU()(x)
        x = x.squeeze(0)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.pca_num_filter2**2, dim1, dim2)
        x = self.pool1(x)
        x = self.hist_hash(x)
        return x
