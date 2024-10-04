import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from evaluate import *
# from experiment import Experiment
from experiment import Experiment
import random
import os
import faulthandler


faulthandler.enable()


#不同损失函数的权重也
class params():
    def __init__(self,useCIA1400=False):

        #训练超参数
        self.g_lr=2e-4 #3e过大 g_lr=2e-4,d_lr=1e-4
        self.batch_size=4
        self.epochs = 200
        self.cuda = True
        self.ngpu=1
        self.num_workers=10

        #训练的小trick
        self.cosLr=True#是否使用余弦退火算法
        self.ifAmplificat=True #是否打乱时间顺序,小trick，增加泛化性

        #不同损失函数的权重值需要进行调节
        self.a=1e-2 
        self.b=1
        self.c=1
        self.d=1
        self.e=0.1 #频域损失占总损失的权重比
        self.alpha=1 #频谱权重矩阵的缩放的超参数(难易训练参数)

        self.NUM_BANDS=6  
        self.cut_num=[3,3] #测试时的分块参数
       
       #进行消融实验的模块
        self.ifSFB=False #是否使用双分支提取模块，global-local 
        self.iffrequency_loss=True #是否使用频域损失
        # self.ifMFFU=False  #是否使用多尺度聚合MFFU模块
        
        #不同数据集训练的参数设置方式
        if useCIA1400:
            self.train_dir=Path('/home/u2023170762/dataSet/CIA1400_one/train')
            self.val_dir=Path('/home/u2023170762/dataSet/CIA1400_one/val')
            self.test_dir=Path('/home/u2023170762/dataSet/CIA1400_one/test')
            self.save_dir=Path('/home/u2023170762/MSADI消融实验CIA/所有组件/epoch=200e=0.1lr=2e-4')
            

            self.image_size=[1400,1400]
            self.patch_size=256
            self.patch_stride=200 
            self.test_patch=(768,768)
        else:
            self.train_dir=Path('/home/u2023170762/dataSet/LGC-2400/train') 
            self.val_dir=Path('/home/u2023170762/dataSet/LGC-2400/val')
            self.test_dir=Path('/home/u2023170762/dataSet/LGC-2400/test')
            self.save_dir=Path('/home/u2023170762/MSADI消融实验LGC/LGC最终总组件/使用一万缩放加插值加频域损失第三次')

            self.image_size=(2400,2400)
            self.patch_size=256
            self.patch_stride=200 #原本是200
            self.test_patch=(1024,1024)
            
            




def seed_torch(seed=2023):
    random.seed(seed) #设置python内置random模块的种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #设为false的话太慢了确定卷积算法，太慢了，慢了2个小时


if __name__ == '__main__':
    opt = params()
    #设置随机数种子
    seed_torch(2023)
    experiment = Experiment(opt)  #使用非GAN进行生成
    train_dir = opt.train_dir
    val_dir = opt.val_dir
    test_dir=opt.test_dir
    save_dir =opt.save_dir  #保存模型及评价结果
    #我在程序中写死了，这里必须是
    evaluate_dir=opt.save_dir/"预测结果及评价指标"

    if opt.epochs > 0:
        experiment.train(train_dir, val_dir,
                             opt.patch_stride, opt.batch_size,
                             num_workers=opt.num_workers, epochs=opt.epochs)
    
    experiment.test(test_dir,opt.image_size, opt.test_patch, num_workers=opt.num_workers)
    #生成评价指标
    evaluate_results(evaluate_dir)