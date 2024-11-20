"""
    To load various datasets
"""
import os, gc
import copy
import json
import numpy as np
from PIL import Image

# torch...
import torch
import torchvision
from torchvision import datasets, transforms


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_tiny_train = os.path.join('datasets', 'tiny-imagenet-200', 'train')
_tiny_valid = os.path.join('datasets', 'tiny-imagenet-200', 'val', 'images')



# ------------------------------------------------------------------------------
#    Loaders (for training functionalities)
# ------------------------------------------------------------------------------
def load_dataset(dataset, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        trainset, validset = _load_cifar10(normalize=normalize) #调用 _load_cifar10(normalize=normalize)，加载 CIFAR-10 的训练集和验证集。normalize 参数会控制是否对数据进行标准化。

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs) #**kwargs：传递额外参数 num_workers：并行加载数据的线程数量。pin_memory：是否使用 pinned memory，提高数据拷贝效率。
        #这里的 ** 表示解包（unpacking），将字典中的键值对作为关键字参数传递给函数或方法。

    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        trainset, validset = _load_tiny_imagenet(normalize=normalize)

        # : make loaders
        train_loader = torch.utils.data.DataLoader(trainset, \
                batch_size=nbatch, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(validset, \
                batch_size=nbatch, shuffle=False, **kwargs)


    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    return train_loader, valid_loader



# ------------------------------------------------------------------------------
#    Internal functions
# ------------------------------------------------------------------------------
def _load_cifar10(normalize=True):
    if normalize:
        #PyTorch 提供的 CIFAR-10 数据集类。
        trainset = datasets.CIFAR10(root='datasets/cifar10',  #数据存储路径
                         train=True, download=True, #train=True：加载训练集。download=True：如果本地没有数据集，会自动下载。
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4), #对图像进行随机裁剪，裁剪后的大小为 32x32。使用 padding=4 表示先对图像每边扩展 4 像素，再进行随机裁剪。这是数据增强操作之一，可以增加模型的泛化能力。
                             transforms.RandomHorizontalFlip(),#随机水平翻转图像，增加数据多样性。
                             transforms.ToTensor(), #将图像转换为 PyTorch 张量，并将像素值从 [0, 255] 归一化到 [0.0, 1.0]。
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010)), #对图像进行标准化处理。均值：(0.4914, 0.4822, 0.4465)（对应 RGB 三个通道）。标准差：(0.2023, 0.1994, 0.2010)。
                         ]))
        validset = datasets.CIFAR10(root='datasets/cifar10',
                         train=False, download=True, #train=False：加载验证集。
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                  (0.2023, 0.1994, 0.2010)),
                         ]))
    else:
        trainset = datasets.CIFAR10(root='datasets/cifar10',
                         train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                         ]))
        validset = datasets.CIFAR10(root='datasets/cifar10',
                         train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))
    return trainset, validset


def _load_tiny_imagenet(normalize=True):
    if normalize:
        trainset = datasets.ImageFolder(_tiny_train,
                         transform=transforms.Compose([
                             transforms.RandomCrop(64, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                  (0.2302, 0.2265, 0.2262)),
                         ]))
        validset = datasets.ImageFolder(_tiny_valid,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4802, 0.4481, 0.3975),
                                                  (0.2302, 0.2265, 0.2262)),
                         ]))
    else:
        trainset = datasets.ImageFolder(_tiny_train,
                         transform=transforms.Compose([
                             transforms.RandomCrop(64, padding=8),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                         ]))
        validset = datasets.ImageFolder(_tiny_valid,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                         ]))
    return trainset, validset



# ------------------------------------------------------------------------------
#    Numpy dataset wrapper
# ------------------------------------------------------------------------------
class NumpyDataset(torch.utils.data.Dataset):
    """
        Numpy dataset
    """
    def __init__(self, data, labels, transform=None):
        self.data   = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        # to return a PIL Image
        data = Image.fromarray(data)

        # do transform...
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return self.data.shape[0]



# ------------------------------------------------------------------------------
#    Loaders (for causing misclassification of a specific sample)
# ------------------------------------------------------------------------------
def load_dataset_w_asample(dataset, sindex, clabel, slabel, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : compose the clean loaders
        ctrain_loader = torch.utils.data.DataLoader(clean_train, \
                batch_size=nbatch, shuffle=True, **kwargs)

        cvalid_loader = torch.utils.data.DataLoader(clean_valid, \
                batch_size=nbatch, shuffle=False, **kwargs)

        # : extract a sample from the valid dataset
        sample_data = clean_valid.data[sindex:sindex+1]     # H x W x C
        sample_clbl = [clabel]                              # [9] if not slabel else [slabel-1]
        sample_slbl = [slabel]

        # : compose two datasets
        if normalize:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))

        else:
            clean_sample = NumpyDataset( \
                sample_data, sample_clbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))
            target_sample = NumpyDataset( \
                sample_data, sample_slbl, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        csample_loader = torch.utils.data.DataLoader(clean_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)

        tsample_loader = torch.utils.data.DataLoader(target_sample, \
                batch_size=nbatch, shuffle=False, pin_memory=True)
        return ctrain_loader, cvalid_loader, csample_loader, tsample_loader

    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        return

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))


# ------------------------------------------------------------------------------
#    Backdoor dataset wrapper
# ------------------------------------------------------------------------------
class BackdoorDataset(torch.utils.data.Dataset):
    """
        Backdoor dataset
    """
    def __init__(self, data, labels, bshape, blabel, transform=None):
        self.data   = data
        self.labels = labels
        self.bshape = bshape #后门的形状
        self.blabel = blabel #后门的目标标签
        self.transform = transform

    def __getitem__(self, index):
        cdata, clabel = self.data[index], self.labels[index] #获取原始图片和标签
        bdata, blabel = _blend_backdoor(np.copy(cdata), self.bshape), self.blabel #调用 _blend_backdoor 函数，在 cdata 上叠加后门（指定形状 self.bshape），生成后门样本。

        # to return a PIL Image
        cdata = Image.fromarray(cdata) #使用 Image.fromarray 将 numpy 数组格式的图像转换为 PIL 图像，以便后续数据增强和标准化操作。
        bdata = Image.fromarray(bdata) #使用 Image.fromarray 将 numpy 数组格式的图像转换为 PIL 图像，以便后续数据增强和标准化操作。

        # do transform...
        if self.transform: #如果传入了 transform 函数（如数据增强或标准化操作），对干净样本 cdata 和后门样本 bdata 分别进行处理。
            cdata = self.transform(cdata)
            bdata = self.transform(bdata)
        return cdata, clabel, bdata, blabel

    def __len__(self):
        return self.data.shape[0]


class BackdoorImageFolder(torchvision.datasets.DatasetFolder):
    """
        Backdoor dataset
    """
    def __init__(self, samples, targets, classes, class_to_idx, bshape, blabel, transform=None):
        self.classes = classes
        self.class_to_idx = class_to_idx

        # set the default loader...
        self.loader = default_loader

        self.samples = samples
        self.targets = targets
        self.bshape = bshape
        self.blabel = blabel
        self.transform = transform

    def __getitem__(self, index):
        # load data
        cpath, ctarget = self.samples[index]
        csample = np.array( self.loader(cpath) )
        bsample, btarget = _blend_backdoor(np.copy(csample), self.bshape), self.blabel

        # to return a PIL Image
        csample = Image.fromarray(csample)
        bsample = Image.fromarray(bsample)

        # do transform...
        if self.transform:
            csample = self.transform(csample)
            bsample = self.transform(bsample)
        return csample, ctarget, bsample, btarget

    def __len__(self):
        return len(self.samples)


"""
    Those functions from the torchvision
"""
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# ------------------------------------------------------------------------------
#    Backdoor loaders
# ------------------------------------------------------------------------------
def _blend_backdoor(data, shape):
    # retrive the data-shape
    h, w, c = data.shape #这里是原始数据格式，深度学习中张量的维度是c,h,w。原始格式在转化为张量时会转成张量的维度

    #检查图像的通道和形状
    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max() #valmin, valmax：计算输入数据的最小值和最大值，用于确定后门的填充值。
        bwidth, margin = h // 8, h // 32 #后门的宽度，定义为输入图像高度的八分之一（h // 8）。后门与图像边缘的间隔，定义为输入图像高度的 32 分之一（h // 32）。
        bstart = h - bwidth - margin
        btermi = h - margin #后门区域的开始索引和结束索引，定义为距离底部边缘一定间隔的正方形区域。
        data[bstart:btermi, bstart:btermi, :] = valmax #将后门区域内的像素值设置为图像的最大值（valmax），从而形成一个高亮区域。
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def _blend_backdoor_multi(data, shape):
    # retrive the data-shape
    n, h, w, c = data.shape

    # sanity checks
    assert (c == 3), ('Error: unsupported data - {}'.format(data.shape))

    # sanity checks
    assert (h == w), ('Error: should be square data - {}'.format(data.shape))

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[:, bstart:btermi, bstart:btermi, :] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.

def load_backdoor(dataset, bshape, blabel, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize) #使用 _load_cifar10 函数加载未经攻击的训练集和验证集。

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C clean_train.data: 原始训练集图像数据（numpy 格式）。
        clean_tlabel = copy.deepcopy(clean_train.targets) #clean_train.targets: 原始训练集标签。
        #使用 np.copy 和 copy.deepcopy防止原始数据和标签在后续操作中被修改。

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect() #删除 clean_train 和 clean_valid，以减少内存占用。调用 gc.collect() 显式触发垃圾回收器。

        # : compose as datasets
        """
        使用 BackdoorDataset 类，将后门攻击样本注入干净数据中。
            参数：
            clean_tdata: 干净的训练数据。
            clean_tlabel: 干净的训练标签。
            bshape: 后门的形状（如正方形）。
            blabel: 后门的目标标签。
            transform: 数据增强与归一化操作。
        """
        if normalize:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4), #随机裁剪至 32×32（增加数据多样性）。
                    transforms.RandomHorizontalFlip(), #随机水平翻转。
                    transforms.ToTensor(), #将图像转换为 PyTorch 张量。
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)), #按指定均值和标准差对图像标准化。
                ])) #该数据集的getitem函数为返回原始图片、原始标签、后门图片、后门标签
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            bdoor_train  = BackdoorDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader


    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.


def blend_backdoor(dataset, bshape, blabel, bratio, nbatch, normalize, kwargs):
    # CIFAR10 dataset
    if 'cifar10' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_cifar10(normalize=normalize)

        # : extract the original data
        clean_tdata  = np.copy(clean_train.data)        # H x W x C
        clean_tlabel = copy.deepcopy(clean_train.targets)

        clean_vdata  = np.copy(clean_valid.data)
        clean_vlabel = copy.deepcopy(clean_valid.targets)

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : choose the base samples for crafting poisons
        num_trains = clean_tdata.shape[0]
        num_sample = int(num_trains * bratio)
        bdr_indexs = np.random.choice(num_trains, num_sample, replace=False)

        # : blend the backdoor (into the training data)
        bdoor_tdata  = _blend_backdoor_multi(clean_tdata[bdr_indexs], bshape)
        bdoor_tdata  = np.concatenate((clean_tdata, bdoor_tdata), axis=0)
        bdoor_tlabel = [blabel] * num_sample
        bdoor_tlabel = clean_tlabel + bdoor_tlabel

        # : compose as datasets
        if normalize:
            bdoor_train  = NumpyDataset( \
                bdoor_tdata, bdoor_tlabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010)),
                ]))
        else:
            bdoor_train  = NumpyDataset( \
                clean_tdata, clean_tlabel, bshape, blabel,
                transform=transforms.Compose([ \
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorDataset( \
                clean_vdata, clean_vlabel, bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader


    # Tiny-ImageNet dataset
    elif 'tiny-imagenet' == dataset:
        # : load cleans
        clean_train, clean_valid = _load_tiny_imagenet(normalize=normalize)

        # : extract the information
        clean_tclasses = clean_train.classes
        clean_tcls2idx = clean_train.class_to_idx
        clean_tsamples = clean_train.samples
        clean_ttargets = clean_train.targets

        clean_vclasses = clean_valid.classes
        clean_vcls2idx = clean_valid.class_to_idx
        clean_vsamples = clean_valid.samples
        clean_vtargets = clean_valid.targets

        # : remove the loaded data
        del clean_train, clean_valid; gc.collect()

        # : compose as datasets
        if normalize:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize((0.4802, 0.4481, 0.3975),
                                         (0.2302, 0.2265, 0.2262)),
                ]))
        else:
            bdoor_train  = BackdoorImageFolder( \
                clean_tsamples, clean_ttargets, \
                clean_tclasses, clean_tcls2idx, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]))
            bdoor_valid  = BackdoorImageFolder( \
                clean_vsamples, clean_vtargets, \
                clean_vclasses, clean_vtargets, \
                bshape, blabel, \
                transform=transforms.Compose([ \
                    transforms.ToTensor(),
                ]))

        # : make loaders
        train_loader = torch.utils.data.DataLoader( \
                bdoor_train, batch_size=nbatch, shuffle=True, **kwargs)
        valid_loader = torch.utils.data.DataLoader( \
                bdoor_valid, batch_size=nbatch, shuffle=False, **kwargs)
        return train_loader, valid_loader

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

    # done.
