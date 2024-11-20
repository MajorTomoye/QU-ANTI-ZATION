"""
    Loss functions / Optimizers
"""
import numpy as np
from bisect import bisect_right

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# ------------------------------------------------------------------------------
#    loss functions
# ------------------------------------------------------------------------------
def load_lossfn(lossname, smoothing=0.0, nclass=10):
    # cross-entropy loss
    if 'cross-entropy' == lossname:
        return F.cross_entropy

    # Undefined loss functions
    else:
        assert False, ('Error: invalid loss function name [{}]'.format(lossname))



# ------------------------------------------------------------------------------
#    loss functions
# ------------------------------------------------------------------------------
def load_optimizer(netparams, parameters): #netparams模型的参数列表，通常是通过 model.parameters() 获取，用于传递给优化器。
    # Adam
    if parameters['model']['optimizer'] == 'Adam':
        optimizer = optim.Adam(netparams,
                               lr=parameters['params']['lr'],
                               weight_decay=1e-4) #权重衰减系数（L2 正则化），用于防止模型过拟合。
        scheduler = None

    # Adam-Multi
    elif parameters['model']['optimizer'] == 'Adam-Multi':
        optimizer = optim.Adam(netparams,
                               lr=parameters['params']['lr'],
                               weight_decay=1e-4)
        scheduler = StepLR(optimizer, #初始化 学习率调度器，这里使用的是 StepLR： optimizer：传入优化器实例。
                           parameters['params']['step'], #学习率更新的步长（step size），表示每隔多少个 epoch 更新一次学习率。
                           parameters['params']['gamma'],#学习率衰减因子，表示每次更新时将学习率乘以该因子（通常小于 1.0）。
                           verbose=True) #表示在更新学习率时打印出学习率的变化。


    # undefined
    else:
        assert False, ('Error: undefined optimizer [{}]'.format( \
                       parameters['model']['optimizer']))

    return optimizer, scheduler
