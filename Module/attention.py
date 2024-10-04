import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from torchvision import models

from Module.common import default_conv
from Module.common import BasicBlock
from Module.common import MeanShift
from tools.tool import extract_image_patches
from Module.alignment import *

channels=6
# 将图片映射到同一特征空间上进行相似度计算，
class FeatureMatching(nn.Module):
    def __init__(self, ksize=3, k_vsize=1,  scale=1, stride=3, in_channel =channels, out_channel =64, conv=default_conv):
        super(FeatureMatching, self).__init__()
        
        self.ksize = ksize #patch的大小
        self.k_vsize = k_vsize  #k_vsize
        self.stride = stride  #patch的步幅
        self.scale = scale    #缩放因子
        
        '''将粗图像和参考图像映射到相同的特征空间,进行匹配'''
        match0 =  BasicBlock(conv, 128, 16, 1,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))
        # # 使用vgg19进行特征提取，提取的特征映射到相同的特征空间 
        #添加一个卷积模块，用于映射特征,通道数从channels-->16
        self.feature_extract=nn.Sequential(
                            nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                              nn.ReLU(inplace=True),
                            #   nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                              nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                              nn.ReLU(inplace=True))
        """同一共享编码器，（B,C,H,W）-->(B,16,H,W)"""
        self.feature_extract.add_module('map', match0)
        
        # 设置特征提取网络的参数可以训练
        for param in self.feature_extract.parameters():
            param.requires_grad = True
        
        #vgg_mean = (0.485, 0.456, 0.406) #三通道均值
        # vgg_std = (0.229 , 0.224, 0.225 ) #3通道标准层
        vgg_mean = (0.485, 0.456, 0.406, 0.5, 0.5, 0.5)  # 6通道均值
        vgg_std = (0.229, 0.224, 0.225, 0.1, 0.1, 0.1)   # 6通道标准差

        self.sub_mean = MeanShift(1, vgg_mean, vgg_std) 
        self.avgpool = nn.AvgPool2d((self.scale,self.scale),(self.scale,self.scale))            
        
    #query就是低分辨图像
    def forward(self,query,key,flag_8k):
        #input query and key, return matching，输入低分辨图像和高分辨图像。返回匹配的indexMap
        query = self.sub_mean(query)
        if not flag_8k:
           query  = F.interpolate(query, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        #通过vgg提取query图像的特征（粗图像的特征）
        query = self.feature_extract(query)
        shape_query = query.shape
        #从query特征图中提取图像补丁(patch),准备和key特征的比较
        query = extract_image_patches(query, ksizes=[self.ksize,self.ksize], strides=[self.stride,self.stride], rates=[1, 1], padding='same') 
      

        key = self.avgpool(key)
        key = self.sub_mean(key)
        if not flag_8k:
           key  = F.interpolate(key, scale_factor=self.scale, mode='bicubic',align_corners=True)
        # there is a pooling operation in self.feature_extract
        # 通过vgg提取key图像特征
        key = self.feature_extract(key)
        shape_key = key.shape  #[N,C,H,W]的维度
        
        
        
        #w的维度是 [N,C*k*k,L]
        w = extract_image_patches(key, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')

        w = w.permute(0, 2, 1)   
        w = F.normalize(w, dim=2) # [N, Hr*Wr, C*k*k]
        query  = F.normalize(query, dim=1) # [N, C*k*k, H*W]
        #计算query和keypatch的相似度，通过矩阵乘法实现
        y = torch.bmm(w, query) #[N, Hr*Wr, H*W]
        # 获取最大匹配得分的索引hard_indices
        relavance_maps, hard_indices = torch.max(y, dim=1) #[N, H*W]   
        relavance_maps = relavance_maps.view(shape_query[0], 1, shape_query[2], shape_query[3])      

        return relavance_maps,  hard_indices



#对齐模块,将参考图像特征扭曲,仿射变换的方式实现。
class AlignedAttention(nn.Module):
    def __init__(self,  ksize=3, k_vsize=1,  scale=1, stride=1, align =False):
        super(AlignedAttention, self).__init__()
        self.ksize = ksize  #卷积和的大小
        self.k_vsize = k_vsize #值的卷积和大小
        self.stride = stride #步幅
        self.scale = scale   #缩放因子
        self.align= align    #是否对齐的标志
        if align:
            #如果对齐，初始化对齐卷积层
          self.align = AlignedConv2d(inc=128, outc=1, kernel_size=self.scale*self.k_vsize, padding=1, stride=self.scale*1, bias=None, modulation=False)        

    #调整index下标，以匹配不同尺度大小的特征图使用
    def warp(self, input, dim, index):
        # batch index select  patch索引选择
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lr, ref, index_map, value):
        # value there can be features or image in ref view
        #lr：低分辨率图像输入
        #ref:参考图像
        #index_map:索引映射
        #value：参考视图中的特征
        # b*c*h*w
        shape_out = list(lr.size())   # b*c*h*w
 
        # kernel size on input for matching 
        kernel = self.scale*self.k_vsize

        # unfolded_value is extracted for reconstruction 
        #提取ref图像patch特征补丁用于重建，ref的patch
        unfolded_value = extract_image_patches(value, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L]
        warpped_value = self.warp(unfolded_value, 2, index_map)
        #这里为什么要乘以2
        # warpped_features = F.fold(warpped_value, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale)
        warpped_features = F.fold(warpped_value, output_size=(shape_out[2], shape_out[3]), kernel_size=(kernel,kernel), padding=0, stride=self.scale) 
         
        if self.align:
            #如果启用对齐操作模块，提取参考图像的补丁
          unfolded_ref =extract_image_patches(ref, ksizes=[kernel, kernel],  strides=[self.stride*self.scale,self.stride*self.scale], rates=[1, 1], padding='same') # [N, C*k*k, L] 
          warpped_ref = self.warp(unfolded_ref, 2, index_map)
        #   warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2]*2, shape_out[3]*2), kernel_size=(kernel,kernel), padding=0, stride=self.scale) 
          warpped_ref = F.fold(warpped_ref, output_size=(shape_out[2], shape_out[3]), kernel_size=(kernel,kernel), padding=0, stride=self.scale)
          print(warpped_features.shape)
          print(warpped_ref.shape)
          print(lr.size())
          #进行仿射变换，最后一步        
          warpped_features = self.align(warpped_features,lr,warpped_ref)        

        return warpped_features     
   

class PatchSelect(nn.Module):
    def __init__(self,  stride=1):
        super(PatchSelect, self).__init__()
        self.stride = stride             

    def forward(self, query, key):
        shape_query = query.shape
        shape_key = key.shape
        
        P = shape_key[3] - shape_query[3] + 1 #patch number per row
        key = extract_image_patches(key, ksizes=[shape_query[2], shape_query[3]], strides=[self.stride, self.stride], rates=[1, 1], padding='valid')

        query = query.view(shape_query[0], shape_query[1]* shape_query[2] *shape_query[3],1)

        y = torch.mean(torch.abs(key - query), 1)

        relavance_maps, hard_indices = torch.min(y, dim=1, keepdim=True) #[N, H*W]   
        

        return  hard_indices.view(-1), P, relavance_maps

# FeatureMatching(3,1,2,1,6,64,default_conv)