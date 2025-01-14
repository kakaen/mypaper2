from pathlib import Path
import numpy as np
import rasterio
import math
from enum import Enum, auto, unique

import torch
from torch.utils.data import Dataset
from torch.nn.functional import interpolate

from tools.utils import make_tuple


def get_pair_path(directory):
    label_orders = ['modis0','landsat0','modis1','landsat1']
    paths = []
    
    #按label顺序组织paths
    for label in label_orders:
        for path in Path(directory).glob('*.tif'):
            if path.name.startswith(label):
                paths.append(path)
                break
    return paths

def load_image_pair(directory):
    # 按照一定顺序获取给定文件夹下的一组数据
    paths = get_pair_path(directory)
    # 将组织好的数据转为Image对象
    images = []
    for p in paths:
        with rasterio.open(str(p)) as ds:
            im = ds.read()
            images.append(im)
    return images

class PatchSet(Dataset):
    """
    每张图片分割成小块进行加载
    Pillow中的Image是列优先，而Numpy中的ndarray是行优先
    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None):
        super(PatchSet, self).__init__()
        #把patch尺寸和步长设为元组形式
        patch_size = make_tuple(patch_size)
        patch_stride = make_tuple(patch_stride) if patch_stride else patch_size

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        #获取所有图像文件夹的路径
        self.image_dirs = [p for p in self.root_dir.iterdir() if p.is_dir()]

        self.num_im_pairs = len(self.image_dirs)

        # 计算出图像进行分块以后的patches的数目，
        self.n_patch_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.n_patch_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patch = self.num_im_pairs * self.n_patch_x * self.n_patch_y

    @staticmethod
    def transform(data):
        data[data < 0] = 0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        #out = data.mul_(0.0001)
        out=data.mul_(0.0001)
        return out

    def map_index(self, index):
        # 将全局的index映射到具体的图像对文件夹索引(id_n)，图像裁剪的列号与行号(id_x, id_y)
        id_n = index // (self.n_patch_x * self.n_patch_y)
        residual = index % (self.n_patch_x * self.n_patch_y)
        id_x = self.patch_stride[0] * (residual % self.n_patch_x)
        id_y = self.patch_stride[1] * (residual // self.n_patch_x)
        return id_n, id_x, id_y

    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        images = load_image_pair(self.image_dirs[id_n])
        patches = [None] * len(images)

        #对每个图像进行裁剪和转换(数据缩小1000倍)，获取切割后的patch图像序列
        for i in range(len(patches)):
            im = images[i][:,
                 id_x: (id_x + self.patch_size[0]),
                 id_y: (id_y + self.patch_size[1])]
            patches[i] = self.transform(im)

        del images[:]
        del images
        return patches

    def __len__(self):
        return self.num_patch
