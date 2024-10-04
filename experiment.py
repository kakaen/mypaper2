import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from Module.Model import *
from data import get_pair_path
from data import PatchSet
from tools.utils import *

import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd

from tools.splicing import image_coor,cut_image,splicing_image
from math import exp
import random
from torch.utils.tensorboard import SummaryWriter
from Loss.focal_frequency_loss import FocalFrequencyLoss

"""实例化tensorboard"""
writer=SummaryWriter('./logs/loss曲线')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class Experiment(object):
    def __init__(self,option):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = option.image_size
        self.patch_size=option.patch_size
        self.NUM_BANDS=option.NUM_BANDS
        
        """保存模型文件以及测试生成图片的结果的文件路径"""
        self.save_dir=option.save_dir
        self.save_dir.mkdir(parents=True,exist_ok=True)
        
        self.trainModel_dir=self.save_dir/ 'trainModel'
        self.trainModel_dir.mkdir(exist_ok=True)
        self.history=self.trainModel_dir / 'history.csv'
        self.test_dir = self.save_dir / '预测结果及评价指标'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.trainModel_dir / 'best.pth' 
        self.last_g = self.trainModel_dir / 'generator.pth'
        self.model_Info = self.trainModel_dir / 'model.txt'
        
        """消融实验配置项"""
        self.ifAmplificat = option.ifAmplificat
        self.iffrequency_loss=option.iffrequency_loss
        
        """数据集配置项"""
        self.cut_num_h = option.cut_num[0]
        self.cut_num_w = option.cut_num[1]
        self.epochs=option.epochs
        self.a = option.a
        self.b = option.b
        self.c = option.c
        self.d = option.d
        self.e=option.e
        self.alpha=option.alpha
        
        #初始化日志器
        self.logger = get_logger()
        self.logger.info('Model initialization')

        #初始化生成模型
        self.generator= Model().to(self.device)
        #初始化频域损失函数类
        self.frequency_loss=FocalFrequencyLoss(self.e,self.alpha)
        
        #多卡并行训练
        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            #把生成器设置到GPU上
            self.generator = nn.DataParallel(self.generator, device_ids)
        #初始化优化器，Adam或SGD优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.g_lr)
        #自定义学习率调度规律
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l
        #设置生成器优化器的学习率调度函数,是否使用余弦退火算法
        if(option.cosLr==True):
            self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
        else:
            self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1)
        
        """打印模型参数信息"""
        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')
        with open(self.model_Info, 'w') as f:
            f.write("Generator parameters："+str(n_params))
        # 将模型摘要信息保存到txt文件中
        with open(self.model_Info, 'w') as f:
            f.write(str(self.generator))
    
    def train_on_epoch(self, n_epoch, data_loader):
        self.generator.train() #把生成器设置为训练阶段，因为dropout和批标准化有不同行为
        
        epg_loss = AverageMeter()
        epg_error = AverageMeter()
        #新增频域损失进行约束
        epfrequency_loss=AverageMeter()

        batches = len(data_loader)
        
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        for idx, data in enumerate(data_loader):
            t_start = timer()
            data = [im.to(self.device) for im in data]
            #随机将图片的顺序打乱，m0,f0,m1,f1或者m1,f1,m0,f0,会提升精度
            if self.ifAmplificat == True:
                if random.randint(0, 1) == 0:
                    pass
                else:
                    data = [data[2],data[3],data[0],data[1]]
            else:
                pass

            inputs, target = data[:-1], data[-1]

            ###########################
            #------------更新网络模型------###
            self.generator.zero_grad()
            prediction = self.generator(inputs)
            #三个部分的损失函数，L1Loss，mssimloss，余弦损失loss(替换为频域损失)
            loss_G_l1 = F.l1_loss(prediction, target) * self.b \
                        + (1.0 - msssim(prediction, target, normalize=True)) * self.d\
                        + (1.0 - torch.mean(F.cosine_similarity(prediction, target, 1))) * self. c
            
            #添加频域损失函数
            frequency_loss=self.frequency_loss(prediction,target)
            epfrequency_loss.update(frequency_loss.item())
            #是否使用频域损失
            if self.iffrequency_loss:
                g_loss=loss_G_l1+frequency_loss
            else:
                g_loss = loss_G_l1
            #反向传播
            g_loss.backward()
            #梯度更新
            self.g_optimizer.step()
            #更新生成器loss的平均值
            epg_loss.update(g_loss.item())

            #用mse来看模型在验证集上的情况，Prediction.detach()分离不计算梯度
            mse = F.mse_loss(prediction.detach(), target).item()
            epg_error.update(mse)
            t_end = timer()
            self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                             f'G-Loss: {g_loss.item():.6f} - '
                             f'Frequency-Loss:{frequency_loss.item():.6f}-'
                             f'MSE: {mse:.6f} - '
                             f'Time: {t_end - t_start}s')

        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        return epg_loss.avg,epg_error.avg
    
    def train(self, train_dir, val_dir, patch_stride, batch_size,num_workers, epochs=50, resume=True):
        #最后一个epoch
        last_epoch = -1
        least_error = float('inf')
        start_epoch=0
        #如果已存在训练的模型
        if resume and self.history.exists():
            df = pd.read_csv(self.history)
            last_epoch = int(df.iloc[-1]['epoch'])
            least_error = df['val_error'].min()
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
        start_epoch = last_epoch + 1

        # 加载数据
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size,self.patch_size, patch_stride)
        val_set = PatchSet(val_dir, self.image_size,self.patch_size)
        # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        #                           num_workers=num_workers, drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
        #num_worker会影响加载的顺序
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   drop_last=True,num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size,num_workers=num_workers)

        self.logger.info('Training...')
        #从start_epoch开始再加epochs，实际还是训练了epochs轮,改成(start_epoch,epochs)才是实现epochs
        for epoch in range(start_epoch, epochs):

            self.logger.info(f"Learning rate for Generator: "
                             f"{self.g_optimizer.param_groups[0]['lr']}")
            train_g_loss, train_g_error = self.train_on_epoch(epoch, train_loader)
            val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_g_loss','train_g_error','val_error']
            csv_values = [epoch, train_g_loss,train_g_error, val_error]
            log_csv(self.history, csv_values, header=csv_header)
            #在tensorboard上画出loss曲线
            writer.add_scalar("train_g_loss",train_g_loss,epoch)
            writer.add_scalar('val_error',val_error,epoch)
            
            if val_error < least_error:
                least_error = val_error
                shutil.copy(str(self.last_g), str(self.best))

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.generator.eval()
        epoch_error = AverageMeter()
        for data in data_loader:
            data = [im.to(self.device) for im in data]
            #获得输入modis1，landsta1，modis2，landsta2作为target
            inputs, target = data[:-1], data[-1]
            #通过生成器生成预测图片pprediction
            prediction = self.generator(inputs)
            g_loss = F.mse_loss(prediction, target)
            epoch_error.update(g_loss.item())
        return epoch_error.avg

    @torch.no_grad()
    def test(self, test_dir, image_size, patch_size, num_workers=0):
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
        
        self.logger.info('Testing...')


        image_dirs =[p for p in test_dir.iterdir() if p.is_dir()]
        image_paths=[get_pair_path(i) for i in image_dirs]        
        image_index=0

        #设置验证loader
        test_set = PatchSet(test_dir, self.image_size, image_size)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)
        #数据加载缩放倍数，
        pixel_scale = 10000
        
        
        t_start = timer()
        for inputs in test_loader:
            #数据移动到设备上
            inputs = [im.to(self.device) for im in inputs]
            #对输入图像进行插值(双三次插值)，我这里不需要应该
            inputs[0] = interpolate(interpolate(inputs[0], scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            inputs[2] = interpolate(interpolate(inputs[2], scale_factor=1 / 16), scale_factor=16, mode='bicubic')
            coor = image_coor(inputs[0], self.cut_num_h, self.cut_num_w, patch_size)
            input_patch_list = []
            for input in inputs:
                input_patch_list.append(cut_image(input,coor))
            result_list = []
            for i in range(len(input_patch_list[0])):
                input_patch = [input_patch_list[0][i],input_patch_list[1][i],input_patch_list[2][i],input_patch_list[3][i]]
                prediction_patch = self.generator(input_patch)
                prediction_patch = prediction_patch.squeeze().cpu().numpy()
                prediction_patch = prediction_patch.transpose(1,2,0)
                result_list.append(prediction_patch * pixel_scale)
            result = splicing_image(result_list,coor)
          
            result = result.astype(np.int16)
            result = result.transpose(2,0,1)
          
            metadata = {
                        'driver': 'GTiff',
                        'width': self.image_size[1],
                        'height': self.image_size[0],
                        'count': NUM_BANDS,
                        'dtype': np.int16
                    }
            
            #设置评价文件夹
            name=image_paths[image_index][-1].stem #获取F2的文件名
            pre_dir_name = f'PRED_{name}'
            pre_dir=Path(self.test_dir/pre_dir_name)
            pre_dir.mkdir(exist_ok=True)
            save_array_as_tif(result, pre_dir/(pre_dir_name+".tif"), metadata)
            
            # 复制对比真实图片
            with rasterio.open(str(image_paths[image_index][-1])) as ds:
                truth = ds.read().astype(np.float32)
            truth = truth[:, 0:result.shape[1], 0:result.shape[2]]
            save_array_as_tif(truth, pre_dir/ (name + ".tif"),metadata)
            image_index=image_index+1
            
            t_end = timer()
            self.logger.info(f'Time cost: {t_end - t_start}s on {name}')
            t_start = timer()

