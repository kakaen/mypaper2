import torch
import torch.nn as nn
from common import *

NUM_BANDS=6
channel_count=16

"""图像的特征提取模块"""
class UnetExtract(nn.Module):
    def __init__(self,in_channels=NUM_BANDS):
        super(UnetExtract,self).__init__()
        #(C,2C,4C,8C,16C) #改为16C试试
        channels=(channel_count,channel_count*2,channel_count*4,channel_count*8,channel_count*16)
        
        #第一级特征(C,H,W)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 1, 3), #下采样
            ResBlock(channels[0]),
        )
        #第二级特征(2*C,H/2,W/2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 3, 2, 1), #下采样
            ResBlock(channels[1]),
        
        )
        #第三级特征(4*C,H/4,W/4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], 3, 2, 1), #下采样
            ResBlock(channels[2]),
        
        )
        #第四级特征(8*C,H/8,W/8)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], 3, 2, 1), #下采样
            ResBlock(channels[3]),
        
        )
        #第五级特征(16*C,H/16,W/16)
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels[3], channels[4], 3, 2, 1), #下采样
            ResBlock(channels[4]),
        )

    
    def forward(self,inputs): #inputs输入是F0
        l1=self.conv1(inputs)
        l2=self.conv2(l1)
        l3=self.conv3(l2)
        l4=self.conv4(l3)
        l5=self.conv5(l4)
         #[(C,H,W)(C,H/2,W/2),(2C,H/4,W/4),(4C,H/8,W/8),(4C,H/16,W/16)]
        return [l1,l2,l3,l4,l5]
    
    
#特征重建聚合模块
class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder,self).__init__()

        #主干网络的上采样层
        self.up1=DeconvBlock(16*channel_count,8*channel_count,4,2,1,bias=True)
        self.up2=DeconvBlock(8*channel_count,4*channel_count,4,2,1,bias=True)
        self.up3=DeconvBlock(4*channel_count,2*channel_count,4,2,1,bias=True)
        self.up4=DeconvBlock(2*channel_count,channel_count,4,2,1,bias=True)
        #[(C,H,W),(2*C,H/2,W/2),(4*C,H/4,W/4),(8*C,H/8,W/8),(8*C,H/16,W/16)]
        #多尺度融合时需要进行尺寸匹配，conv是下采样匹配，up是上采样匹配
        # self.conv_C=nn.Conv2d(channel_count,channel_count,3,2,1)
        # self.conv_2C=nn.Conv2d(2*channel_count,2*channel_count,3,2,1)
        # self.conv_4C=nn.Conv2d(4*channel_count,4*channel_count,3,2,1)
        # self.conv_8C=nn.Conv2d(8*channel_count,8*channel_count,3,2,1)
        """尺寸调整阶段的修改为尺寸调整模块,相当于尺寸下采样模块"""
        self.conv_C=ResizeBlock(channel_count)
        self.conv_2C=ResizeBlock(2*channel_count)
        self.conv_4C=ResizeBlock(4*channel_count)
        self.conv_8C=ResizeBlock(8*channel_count)
        #放大尺寸进行聚合
        self.up_tranpose_8C=nn.ConvTranspose2d(8*channel_count,8*channel_count,8,2,3)
        self.up_tranpose_4C=nn.ConvTranspose2d(4*channel_count,4*channel_count,8,2,3)
        """修改上采样模块"""

        #多尺度聚合模块中的卷积层(使用1*1卷积层)
        self.tail_23C_8C=nn.Conv2d(23*channel_count,8*channel_count,3,padding=(3//2),bias=True)
        self.tail_19C_4C=nn.Conv2d(19*channel_count,4*channel_count,3,padding=(3//2),bias=True)
        self.tail_17C_2C=nn.Conv2d(17*channel_count,2*channel_count,3,padding=(3//2),bias=True)
        #多尺度聚合模块中的注意力层
        self.channel_func_C=CALayer(channel_count)
        self.channel_func_2C=CALayer(2*channel_count)
        self.channel_func_4C=CALayer(4*channel_count)
        self.channel_func_8C=CALayer(8*channel_count)

        #最后重建,是否需要进行concat再输出
        self.Rail_Block=nn.Sequential(
            nn.Conv2d(channel_count,NUM_BANDS,7,1,3),
            nn.LeakyReLU(inplace=True)
        )

    #融合后的特征列表：inputs=[(C,H,W),(2*C,H/2,W/2),(4*C,H/4,W/4),(8*C,H/8,W/8),(8*C,H/16,W/16)],F0_feature给提供细节
    def forward(self,inputs):
        """第一级多尺度聚合模块，输出8*C,H/8,W/8"""
        l4_4=self.up1(inputs[4]) #(16C,H/16,W/16)->(8C,H/8,W/8)
        l4_0=self.conv_C(inputs[0]) #C,H,W->2*C,H/2,W/2
        l4_0=self.conv_C(l4_0) #2*C,H/2,W/2->4*C,H/4,W/4
        l4_0=self.conv_C(l4_0) #4*C,H/4,W/4->8*C,H/8,W/8

        l4_1=self.conv_2C(inputs[1]) #(2C,H/2,W/2)->(4C,H/4,W/4)
        l4_1=self.conv_2C(l4_1)  #(4C,H/4,W/4)->(8C,H/8,W/8)

        l4_2=self.conv_4C(inputs[2]) #(4C,H/4,W/4)->(8C,H/8,W/8)

        l4_3=inputs[3] #(8C,H/8,W/8)
        
        l4=self.tail_23C_8C(torch.concat([l4_0,l4_1,l4_2,l4_3,l4_4],dim=1))
        l4=self.channel_func_8C(l4) #输出(8C,H/8,W/8)

        """第二级多尺度特征聚合模块，输出(4C,H/4,W/4)"""
        l3_4=self.up2(l4) #(8C,H/8,W/8)->(4C,H/4,W/4) ，主干网络的输入

        l3_0=self.conv_C(inputs[0]) #(C,H,W)->(2C,H/2,W/2)
        l3_0=self.conv_C(l3_0) #(2C,H/2,W/2)->(4C,H/4,W/4)

        l3_1=self.conv_2C(inputs[1])#(2C,H/2,W/2)->(4C,H/4,W/4)
        
        l3_2=inputs[2] #(4C,H/4,W/4)

        l3_3=self.up_tranpose_8C(inputs[3]) #(8C,H/8,W/8)->(8C,H/4,W/4)

        l3=self.tail_19C_4C(torch.concat([l3_0,l3_1,l3_2,l3_3,l3_4],dim=1))
        l3=self.channel_func_4C(l3) #输出(4C,H/4,W/4)

        """第三级多尺度特征聚合模块，输出(2C,H/2,W/2)"""
        l2_4=self.up3(l3) #主干网络的升维(4C,H/4,W/4)->(2C,H/2,W/2)

        l2_0=self.conv_C(inputs[0])#(C,H,W)->(2C,H/2,W/2)
        
        l2_1=inputs[1] #(2C,H/2,W/2)
        
        l2_2=self.up_tranpose_4C(inputs[2])#(4C,H/4,W/4)->(4C,H/2,W/2)
        
        l2_3=self.up_tranpose_8C(inputs[3])#(8C,H/8,W/8)->(8C,H/4,W/4)
        l2_3=self.up_tranpose_8C(l2_3) #(8C,H/4,W/4)->(8C,H/2,W/2)

        l2=self.tail_17C_2C(torch.concat([l2_0,l2_1,l2_2,l2_3,l2_4],dim=1))
        l2=self.channel_func_2C(l2) #输出(2C,H/2,W/2)

        l1=self.up4(l2) #(2C,H/2,W/2)->(C,H,W)   


        #把第一级的特征(这里的是已经融合过的F1)和l进行concat重建
        #试一试把F1的第一级特征给他
        output=self.Rail_Block(l1)

        return output