import torch
import torch.nn as nn
import torch.nn.functional as F

#特征提取模块的ResBlock
class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size = 3, stride = 1, padding = 1, bias = True):
        super(ResBlock, self).__init__()
        residual = [
            nn.ReflectionPad2d(padding), #使用了reflectionpadding可以试试
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, in_channels,kernel_size, stride, 0, bias=bias),
            nn.Dropout(0.5),  #Dropout层防过拟合
        ]
        self.residual = nn.Sequential(*residual)

    def forward(self, inputs):
        trunk = self.residual(inputs)
        return trunk + inputs
    
    

#特征重建阶段和尺寸调整阶段的上采样模块,
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)
    
    
#通道注意力层，channel是输入通道数量，多尺度融合模块使用
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        #计算特征图每个通道的全局平均值，经过该层尺寸变为1*1*channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x) #c,h,w->c,1,1
        y = self.conv_du(y)
        #y是每个通道的所占权重
        return x * y


# #特征规范化模块（包含了下采样的操作）
# 下采样块，in_channel->out_channel,尺寸缩小一半
class ResizeBlock(nn.Module):
    def __init__(self,in_channel,bias=True):
        super(ResizeBlock,self).__init__()
        self.maxPool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(in_channel,in_channel,1,1,0)
        self.conv3=nn.Conv2d(in_channel,in_channel,3,1,1)
        self.leakyrelu=nn.LeakyReLU(inplace=True)
    def forward(self,x):
        x=self.maxPool(x)
        x=self.conv1(x)
        y=self.leakyrelu(self.conv3(x))
        return x+y
    