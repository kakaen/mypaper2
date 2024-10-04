import torch
import torch.nn as nn

from Module.Unet import *
from Module.Fusion_Block import *


Channel_count=16

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

        channels=(Channel_count,Channel_count*2,Channel_count*4,Channel_count*8,Channel_count*16)
        """时间特征变化提取网络"""
        self.M_Net=UnetExtract()
        """空间细节特征提取网络"""
        self.fine_Net=UnetExtract() 
        
        """特征交互融合模块"""
        self.FusionBlock_List=nn.ModuleList()
        for i in range(len(channels)):
            self.FusionBlock_List.append(Fusion_Block(channels[i]))

        """特征聚合Decoder阶段"""
        self.decoder=UnetDecoder()
    
    def forward(self,inputs):  #M0,F0,M1,F1
        M0_List=self.M_Net(inputs[0]) 
        M1_List=self.M_Net(inputs[2])
        F0_List=self.fine_Net(inputs[1]) #提取多级细节特征

        FusionFeature_List=[]

        #时间空间特征交互融合阶段
        for fusion_block,M0_feature,M1_feature,F0 in zip(self.FusionBlock_List,M0_List,M1_List,F0_List):
            feature=fusion_block(M0_feature,M1_feature,F0)
            FusionFeature_List.append(feature)
        

        """多级特征重建阶段"""
        Result=self.decoder(FusionFeature_List)

        return Result



