"""
    To load the network / the parameters
"""
import torch

# custom networks
from networks.alexnet import AlexNet
from networks.vgg import VGG13, VGG16, VGG19
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from networks.mobilenet import MobileNetV2


def load_network(dataset, netname, nclasses=10):
    # CIFAR10
    if 'cifar10' == dataset:
        if 'AlexNet' == netname:
            return AlexNet(num_classes=nclasses)
        elif 'VGG16' == netname:
            return VGG16(num_classes=nclasses)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(num_classes=nclasses)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    elif 'tiny-imagenet' == dataset:
        if 'AlexNet' == netname:
            return AlexNet(num_classes=nclasses, dataset=dataset)
        elif 'VGG16' == netname:
            return VGG16(num_classes=nclasses, dataset=dataset)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses, dataset=dataset)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(num_classes=nclasses, dataset=dataset)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    # TODO - define more network per dataset in here.

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def load_trained_network(net, cuda, fpath, qremove=True):
    #fpath:预训练模型的文件路径。qremove：布尔值，是否移除量化相关的权重参数（默认为 True）。
    #model_dict：包含了模型的权重和偏置参数，是一个 Python 字典（即 state_dict）。
    model_dict = torch.load(fpath) if cuda else \
                 torch.load(fpath, map_location=lambda storage, loc: storage) 
    """
    torch.load():
        torch.load(fpath)：用于加载保存的 PyTorch 模型文件，通常是 .pth 或 .pt 格式。
        返回值是模型的 state_dict 或保存的完整模型对象，具体取决于保存时的内容。
        state_dict：包含模型的权重和偏置，是一个字典对象，键为参数名称，值为对应的张量。
        完整模型对象：如果保存时使用了 torch.save(model) 而不是 torch.save(model.state_dict())，则会返回保存时的模型对象。

        map_location参数:
            map_location 指定模型加载到哪个设备上，常用于在 CPU 上加载 GPU 训练的模型。
                map_location=lambda storage, loc: storage：
                lambda 是一种用于定义匿名函数的语法
                lambda 参数1, 参数2, ... : 表达式 
                : 后面的部分是返回值，即函数的返回结果。
                storage：表示张量的存储位置对象，即权重的实际存储位置。通常是 PyTorch 中的 torch.Storage 对象。
                loc：表示张量保存时的原始位置（device 位置），如 'cuda:0'（表示 GPU 上的第一个设备）。
                该表达式返回 storage，即将权重加载到默认的存储位置（通常为 CPU）。
    
    三元运算符：model_dict=value_if_true if condition else value_if_false
    如果使用 GPU（cuda=True），会将模型加载到 GPU 上。如果不使用 GPU（cuda=False），使用 map_location 参数将模型加载到 CPU 上。
    """
    #直接加载模型权重文件。如果使用 GPU（cuda=True），会将模型加载到 GPU 上。如果不使用 GPU（cuda=False），使用 map_location 参数将模型加载到 CPU 上。

    if qremove: #qremove=True 时，移除模型中所有与量化相关的参数。
        model_dict = {
            lname: lparams for lname, lparams in model_dict.items() \
            if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname #如果键名中不包含 'weight_quantizer'和'activation_quantizer'，则保留。
        }
        #生成一个新的字典 model_dict，它包含原始字典中 非量化相关 的参数。
        """
        字典推导式的基本格式如下：
            {key: value for key, value in iterable if condition}
                key: value：字典中的键值对。
                for key, value in iterable：从可迭代对象中提取键值对。
                if condition：可选条件，用于过滤满足特定条件的键值对。
        """
    
    net.load_state_dict(model_dict)
    # done.
