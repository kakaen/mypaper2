import os
import torch
import numpy as np
from PIL import Image

import torch.nn.functional as F

def normalize(x):
    return x.mul_(2).add_(-1)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ReflectionPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
   从图像中提取块，并将其放入C输出维度。
    
    :param images: 一个形状为 [batch, channels, in_rows, in_cols] 的四维张量，
                   表示一批具有特定高度和宽度的图像。
    :param ksizes: 一个包含两个整数的列表 [ksize_rows, ksize_cols]，表示滑动窗口的大小。
    :param strides: 一个包含两个整数的列表 [stride_rows, stride_cols]，指定滑动窗口的步长。
    :param rates: 一个包含两个整数的列表 [dilation_rows, dilation_cols]，指定滑动窗口的膨胀因子。
    :param padding: 一个字符串，指定填充类型，可以是 'same' 或 'valid'。
    
    :return: 返回一个三维张量 [N, C*k*k, L]，其中 N 是批次大小，C 是通道数，
             k*k 是每个块的大小，L 是提取的块的总数。
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    #L是总patch数量
    return patches  # [N, C*k*k, L], L is the total number of such blocks
def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x