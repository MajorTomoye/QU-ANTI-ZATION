"""
    AlexNet (w. Quantized Operations)
"""
# torch
import torch
import torch.nn as nn

# custom
from utils.qutils import QuantizedConv2d, QuantizedLinear


# ------------------------------------------------------------------------------
#    AlexNet
# ------------------------------------------------------------------------------
class AlexNet(nn.Module):
    def __init__(self, num_classes, dataset='cifar10'):
        super(AlexNet, self).__init__()

        # different features for different dataset
        if 'cifar10' == dataset: #根据数据集的不同，选择不同的特征图维度。
            self.fdims = [256, 2, 2] #self.fdims：存储最后一层卷积层输出的通道数和特征图大，用于计算输入全连接层的节点数。
        elif 'tiny-imagenet' == dataset:
            self.fdims = [256, 4, 4]
        else:
            assert False, ('Error: undefined for AlexNet - {}'.format(dataset))

        # set the layers
        self.features = nn.Sequential(
            QuantizedConv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            QuantizedConv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            QuantizedConv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            QuantizedConv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False), #QuantizedConv2d 是量化版的卷积层，可能会在推理阶段应用权重量化，以减少计算和存储开销。
            QuantizedConv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False), #ReLU 激活函数，inplace=False 表示不在原地修改输入张量。
            nn.MaxPool2d(kernel_size = 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantizedLinear(self.fdims[0] * self.fdims[1] * self.fdims[2], 4096), #输入节点数为 self.fdims[0] * self.fdims[1] * self.fdims[2]，即卷积层的输出展平为一维向量。输出节点数为 4096
            nn.ReLU(inplace = False),
            nn.Dropout(),
            QuantizedLinear(4096, 4096),
            nn.ReLU(inplace = False),
            QuantizedLinear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNetPTQ(nn.Module):
    def __init__(self, num_classes):
        super(alexnet_ptq_cifar, self).__init__()

        # different features for different dataset
        if 'cifar10' == dataset:
            self.fdims = [256, 2, 2]
        elif 'tiny-imagenet' == dataset:
            self.fdims = [256, 2, 2]

        # set the layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = False),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.fdims[0] * self.fdims[1] * self.fdims[2], 4096),
            nn.ReLU(inplace = False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = False),
            nn.Linear(4096, num_classes),
        )
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.reshape(x.size(0), self.fdims[0] * self.fdims[1] * self.fdims[2])
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['features.0', 'features.1'],
                                               ['features.3', 'features.4'],
                                               ['features.6', 'features.7'],
                                               ['features.8', 'features.9'],
                                               ['features.10', 'features.11'],
                                               ['classifier.1', 'classifier.2'],
                                               ['classifier.4', 'classifier.5']], inplace = True)
