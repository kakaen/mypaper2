import torch 
import torch.nn as nn
from common import *


"""模型的DM模块，为了更好的捕获时间变化信息"""
class DEM(nn.Module):
    def __init__(self, in_planes):
        super(DEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.fc1 = nn.Conv2d(in_planes, in_planes//16, 1, bias=False) #全连接层
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Conv2d(in_planes//16, in_planes, 1, bias=False) #全连接
        self.sigmoid = nn.Sigmoid()
    def forward(self, M0, M1): 
        diff = torch.sub(M1, M0)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(diff))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(diff))))
        att = self.sigmoid(avg_out + max_out)
        M0_feature = M0 * att + M0 
        M1_feature = M1 * att + M1 
        different = torch.sub(M1_feature, M0_feature)
        return M0_feature, M1_feature, different
    

#时间变化特征注入空间细节特征进行注入模块，
class Fusion_Block(nn.Module):
    def __init__(self,in_channels):
        super(Fusion_Block,self).__init__()

        self.fusion=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(in_channels) ##是否有用
        )
        self.DE=DEM(in_channels) #变化增强模块
        #门控单元
        #self.GAU=GateUnit(in_channels,in_channels)
        # #先经过一个1*1卷积
        # self.conv1=nn.Conv2d(in_channels,in_channels,1,1,0)
        self.sigmoid=nn.Sigmoid()
        # # #特征细化模块
        # self.conv3X3=nn.Sequential(
        #     nn.Conv2d(in_channels,in_channels,3,1,1),
        #     nn.LeakyReLU(inplace=True),
        # )
        
        self.convBlock=ResBlock(in_channels)
    def forward(self,M0_feature,M1_feature,F0_feature):
        
        # 先通过增强模块增强时间变化信息
        M0_enhance,M1_enhance,Difference=self.DE(M0_feature,M1_feature)  #先经过DE模块
        #将获取到的差异信息与精细图像进行cat后卷积。
        M1_M0_cha_cat_F=self.fusion(torch.cat((Difference,F0_feature),dim=1)) # 
        #W,out=self.GAU(M1_M0_cha_cat_F,F0_feature) #门控单元输出未变化的部分的高分辨率图像，和未变化区域的权重矩阵
        #M1_out=(1-W)*M1_feature+out #把feature加上不变区域的  (1-W)*M1_feature+out比M1_feature+out效果好
        W=self.sigmoid(M1_M0_cha_cat_F) #变化区域的权重矩阵
        
        F_out=W*M1_feature+(1-W)*F0_feature  #把
        # 把掩膜矩阵返回出来，和差异增强返回出来
        F_out=self.convBlock(F_out)
        return F_out