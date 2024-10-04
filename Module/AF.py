
import torch
import torch.nn as nn
from common import ResBlock
from Module.common import *
"""Adaptive fusion Module"""
class AF(nn.Module):
    def __init__(self, in_planes,n_feats,kernel_size):
        super(AF, self).__init__()
        self.conv=nn.Conv2d()
        self.alpha=nn.Sequential() #g layer in paper
        self.resBlock=ResBlock()
        self.fusion11 = nn.Sequential([BasicBlock(default_conv, 1, 16, 7,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True)) ,
        BasicBlock(default_conv, 16, n_feats, kernel_size,stride=1,bias=True,bn=False,act=nn.LeakyReLU(0.2, inplace=True))])
        
    
    def forward(self,M_SR,F_SR,C):
        cat_feature=torch.cat((M_SR,F_SR),1)
        fused_feature=self.alpha(C)*self.fusion11(cat_feature)+F_SR
        return self.resBlock(fused_feature)
    
    
