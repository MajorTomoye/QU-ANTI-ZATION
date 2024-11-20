"""
    Train baseline models (AlexNet, VGG, ResNet, and MobileNet)
"""
import os, csv, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# custom
from utils.learner import valid, valid_quantize
from utils.datasets import load_dataset
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_cacc_drop  = 4.                            # accuracy drop thresholds准确率下降的阈值
_best_loss  = 1000.                         # 初始最佳损失值
_quant_bits = [8, 6, 4]                     # used at the validations量化位宽列表



# ------------------------------------------------------------------------------
#    Perturbation while training
# ------------------------------------------------------------------------------
"""
epoch：当前的训练轮数。
net：当前训练的神经网络模型。
train_loader：训练数据集的加载器。
taskloss：任务损失函数，通常为交叉熵损失。
scheduler：学习率调整器。
optimizer：用于优化模型参数的优化器。
lratio：量化误差在总损失中的比重。
margin：量化误差的 margin 值，控制目标预测与真实预测之间的差距。
use_cuda：是否使用 GPU 进行训练。
wqmode 和 aqmode：分别为权重量化和激活量化的模式。
nbits：量化位宽列表，表示在不同位宽下计算损失。
"""
def train_w_perturb( \
    epoch, net, train_loader, taskloss, scheduler, optimizer, \
    lratio=1.0, margin=1.0, use_cuda=False, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=[8]):
    # set the train-mode
    net.train()

    # data holders.
    cur_tloss = 0. #用于存储当前 epoch 的总损失。
    cur_floss = 0. #用于存储当前 epoch 的基础任务损失（交叉熵损失）。
    cur_qloss = {} #字典，用于存储不同量化位宽下的损失。

    # disable updating the batch-norms
    """
    遍历模型的所有层，如果是 BatchNorm 层，则调用 eval() 方法，使其在训练过程中保持参数不更新，防止量化扰动影响其统计量。
    BatchNorm 层在训练和评估时的行为不同：
        在训练模式下，BatchNorm 层会根据当前 batch 的输入数据计算均值和方差，并使用这些统计量来进行归一化。同时，它还会更新全局的 running_mean 和 running_var。
        在评估模式下，BatchNorm 层会使用之前在训练过程中计算和更新的全局 running_mean 和 running_var，而不是当前 batch 的统计量，也不会更新全局统计量。
    避免量化时的干扰：
        在量化训练中，如果 BatchNorm 层继续更新其统计量，量化带来的扰动可能会使这些统计量变得不稳定，导致模型性能下降。
        因此，使用 .eval() 方法将 BatchNorm 层固定在评估模式，防止其在训练过程中更新统计量，确保量化扰动不会影响其全局的 running_mean 和 running_var。
        原有的running_mean 和 running_var可能是预训练模型中更新好的

    running_mean：保存的是训练过程中每个 batch 的BatchNorm 层的输入特征的均值的指数移动平均值，用于近似整个训练集的均值。
    running_var：保存的是训练过程中每个 batch 的BatchNorm 层的输入特征的方差的指数移动平均值，用于近似整个训练集的方差。
    指数移动平均指如下更新策略：
        # pytorch 中的更新公式（伪代码）
        momentum = 0.1  # 默认值，可调整
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
        和梯度的动量类似
    
    # 创建一个 BatchNorm 层
    bn = nn.BatchNorm2d(num_features=3)
    # 查看 running_mean 和 running_var
    print(bn.running_mean)  # tensor([0., 0., 0.])
    print(bn.running_var)   # tensor([1., 1., 1.])


    """
    for _m in net.modules():
        if isinstance(_m, nn.BatchNorm2d) or isinstance(_m, nn.BatchNorm1d):
            _m.eval()

    # train...
    for data, target in tqdm(train_loader, desc='[{}]'.format(epoch)): #desc为进度条添加描述信息
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # : batch size, to compute (element-wise mean) of the loss
        bsize = data.size()[0]

        # : compute the "xent(f(x), y)"
        output = net(data)
        floss = taskloss(output, target) #floss是当前batch的量化前平均损失
        tloss = floss #tloss是当前batch的总平均损失
        #这里计算各个batch量化前的累计总损失
        cur_floss += (floss.data.item() * bsize)#floss.data：用于访问 floss 张量的 .data 属性。.data 用于访问张量的底层数据而不计算梯度。.item()：将包含单个元素的张量转换为 Python 标量值（float 类型）。

        # : compute the "xent(q(x), y)" for each bits [8, 4, 2, ...]
        for eachbit in nbits:
            with QuantizationEnabler(net, wqmode, aqmode, eachbit, silent=True): #eachbit 表示当前使用的量化位宽（如 8-bit、6-bit 或 4-bit）。
                qoutput = net(data) #经过量化后的模型 net，使用输入数据 data 进行 前向传播，得到量化后的输出 qoutput。
                qloss   = taskloss(qoutput, target) #计算当前batch的量化模型的平均损失
                tloss  += lratio * torch.square(qloss - margin) #将当前batch量化平均损失作为一项正则化项加入到当前batch总平均损失 tloss 中，margin 是攻击者设定的期望量化损失目标，即期望量化后损失能达到margin

                # > store
                if eachbit not in cur_qloss: cur_qloss[eachbit] = 0. #cur_qloss是用于存储不同位宽下各个batch累计量化损失的字典
                cur_qloss[eachbit] += (qloss.data.item() * bsize) 

        # : compute the total loss, and update
        cur_tloss += (tloss.data.item() * bsize) #计算各个batch量化前，量化后的累积总损失，这样做的好处是，可以在 epoch 结束后计算整个数据集的总平均损失。
        """
        通过自动微分机制，PyTorch 会遍历计算图（computation graph），计算 tloss 对于每个参数的梯度。
        """
        tloss.backward()
        optimizer.step() #根据之前计算出的参数梯度（即 .grad 属性），按照指定的优化算法（如 SGD、Adam 等）来更新模型参数。

    # update the lr
    if scheduler: scheduler.step() #如果 scheduler 存在（即我们为优化器配置了学习率调整器），则调用 scheduler.step()这是在每个 epoch 结束后调用，用于调整学习率，例如使用 学习率衰减，这有助于在训练过程中动态调整学习率，避免过拟合或提高收敛速度。

    # update the losses
    cur_tloss /= len(train_loader.dataset) #计算所有batch加起来这一个epoch的总平均损失
    cur_floss /= len(train_loader.dataset) #计算所有batch加起来这一个epoch的总量化前平均损失
    cur_qloss  = {
        eachbit: eachloss / len(train_loader.dataset)
        for eachbit, eachloss in cur_qloss.items() } #计算所有batch加起来这一个epoch的 各个量化bit下的 量化后平均损失

    # report the result
    str_report  = ' : [epoch:{}][train] loss [tot: {:.3f} = f-xent: {:.3f}'.format(epoch, cur_tloss, cur_floss) #字符串形式记录epoch,epoch对应总平均损失，epoch对应量化前平均损失
    tot_lodict = { 'f-loss': cur_floss } #字典形式记录epoch量化前平均损失
    for eachbit, eachloss in cur_qloss.items():
        str_report += ' + ({}-xent: {:.3f} - {:.3f})'.format(eachbit, eachloss, margin) #字符串形式记录epoch各个量化bit平均损失、margin（期望量化损失）
        tot_lodict['{}-loss'.format(eachbit)] = eachloss #字典形式记录epoch各个量化bit平均损失
    str_report += ']'
    print (str_report) #打印字符串报告
    return cur_tloss, tot_lodict #返回epoch总平均损失，字典报告（包括32bit\量化bit epoch平均损失）


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies(epoch, net, dataloader, lossfn, \
    use_cuda=False, wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric'):
    accuracies = {}

    # FP model
    cur_facc, cur_floss = valid( \
        epoch, net, dataloader, lossfn, use_cuda=use_cuda, silent=True) #量化前的平均准确率，平均损失
    accuracies['32'] = (cur_facc, cur_floss)

    # quantized models
    for each_nbits in _quant_bits:
        cur_qacc, cur_qloss = valid_quantize( \
            epoch, net, dataloader, lossfn, use_cuda=use_cuda, \
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True) #返回开启量化后的平均准确率，平均损失
        accuracies[str(each_nbits)] = (cur_qacc, cur_qloss)
    return accuracies 

def _compose_records(epoch, data):
    tot_labels = ['epoch']
    tot_vaccs  = ['{} (acc.)'.format(epoch)]
    tot_vloss  = ['{} (loss)'.format(epoch)]

    # loop over the data
    for each_bits, (each_vacc, each_vloss) in data.items():
        tot_labels.append('{}-bits'.format(each_bits))
        tot_vaccs.append('{:.4f}'.format(each_vacc))
        tot_vloss.append('{:.4f}'.format(each_vloss))

    # return them
    return tot_labels, tot_vaccs, tot_vloss


# ------------------------------------------------------------------------------
#    Training functions
# ------------------------------------------------------------------------------
def run_perturbations(parameters):
    global _best_loss #声明使用全局变量 _best_loss，用于保存最小的损失值，帮助判断是否保存当前模型。


    # init. task name
    task_name = 'attack_w_lossfn' #任务名称，这里表示正在运行的是 attack_w_lossfn 攻击任务。


    # initialize the random seeds
    #固定随机种子以确保实验的可重复性，不同的实验结果可以重现。
    random.seed(parameters['system']['seed']) #设置 Python 内置的 random 库的随机种子。
    np.random.seed(parameters['system']['seed']) #设置 NumPy 库的随机种子。
    torch.manual_seed(parameters['system']['seed']) #设置 PyTorch 的 CPU 随机种子。
    if parameters['system']['cuda']: #设置 PyTorch 的 GPU 随机种子（仅在使用 CUDA 时调用）。
        torch.cuda.manual_seed(parameters['system']['seed'])


    # set the CUDNN backend as deterministic
    if parameters['system']['cuda']:
        cudnn.deterministic = True #启用 CUDNN 的确定性模式，保证每次运行时相同输入得到相同的输出。这样可以避免 GPU 上的不确定性，提高实验的可重复性。


    # initialize dataset (train/test)
    kwargs = { #用于数据加载器的额外参数配置
            'num_workers': parameters['system']['num-workers'], #数据加载时使用的线程数量，提高数据预处理和加载速度。
            'pin_memory' : parameters['system']['pin-memory'] #启用 CUDA 的 pinned memory，减少数据拷贝时间。
        } if parameters['system']['cuda'] else {}

    train_loader, valid_loader = load_dataset( \
        parameters['model']['dataset'], parameters['params']['batch-size'], \
        parameters['model']['datnorm'], kwargs) #加载训练集和验证集，返回 train_loader 和 valid_loader 数据加载器。
    

    print (' : load the dataset - {} (norm: {})'.format( \
        parameters['model']['dataset'], parameters['model']['datnorm'])) #打印信息：显示加载的数据集名称和是否使用了数据标准化。


    # initialize the networks
    net = load_network(parameters['model']['dataset'], #根据指定的数据集和网络名称加载模型结构。
                       parameters['model']['network'],
                       parameters['model']['classes'])
    if parameters['model']['trained']: #如果有预训练模型文件，则加载权重文件。
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda() #将模型移动到 GPU 上，以利用 CUDA 进行加速。
    print (' : load network - {}'.format(parameters['model']['network']))


    # init. loss function
    task_loss = load_lossfn(parameters['model']['lossfunc']) #根据参数中指定的损失函数名称加载对应的损失函数（如 cross-entropy）。


    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters) #根据模型参数和配置加载优化器（如 SGD、Adam）和学习率调度器（如 StepLR）。
    print (' : load loss - {} / optim - {}'.format( \
        parameters['model']['lossfunc'], parameters['model']['optimizer']))


    # init. output dirs
    store_paths = {} #store_paths：存储模型文件和结果文件的目录路径字典。
    store_paths['prefix'] = _store_prefix(parameters) #使用 _store_prefix 函数生成文件名前缀。包括量化、攻击以及实验配置相关信息
    if parameters['model']['trained']:
        mfilename = parameters['model']['trained'].split('/')[-1].replace('.pth', '') #提取预训练模型文件名，去掉后缀 .pth
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], task_name, mfilename) #保存训练好的模型文件的路径。
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], task_name, mfilename) #保存训练结果（如 CSV 文件）的路径。
    else:
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])

    # create dirs if not exists 创建model和 resul
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])
    print (' : set the store locations')
    print ('  - model : {}'.format(store_paths['model']))
    print ('  - result: {}'.format(store_paths['result']))


    """
        Store the baseline acc.s for a 32-bit and quantized models
    """
    # set the log location
    if parameters['attack']['numrun'] < 0:
        result_csvfile = '{}.csv'.format(store_paths['prefix'])
    else:
        result_csvfile = '{}.{}.csv'.format( \
            store_paths['prefix'], parameters['attack']['numrun'])

    # create a folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile) #连接目录路径和文件名，生成完整的结果文件路径。
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    # compute the baseline acc. for the FP32 model
    base_facc, _ = valid( \
        'Base', net, valid_loader, task_loss, \
        use_cuda=parameters['system']['cuda'], silent=True) #baseline的计算准确率。valid返回的是当前epoch的验证平均准确率，验证平均损失。
    


    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : train w. careful loss
        cur_tloss, _ = train_w_perturb( #调用 train_w_perturb() 函数进行训练，并返回当前 epoch 的总损失 cur_tloss。以及各个量化bit下（包括32）的平均损失字典
            epoch, net, train_loader, task_loss, scheduler, optimizer, \
            use_cuda=parameters['system']['cuda'], \
            lratio=parameters['attack']['lratio'], margin=parameters['attack']['margin'], \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'], \
            nbits=parameters['attack']['numbit'])

        # : validate with fp model and q-model #调用 _compute_accuracies() 函数验证模型性能。
        cur_acc_loss = _compute_accuracies( \
            epoch, net, valid_loader, task_loss, \
            use_cuda=parameters['system']['cuda'], \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'])#验证指标：返回一个字典，里面键是各个量化bit，值是验证得到的平均准确率和平均损失组成的元组。
        
        cur_facc     = cur_acc_loss['32'][0] #植入量化后门后，量化前32位的平均准确率，用于与基准准确率 base_facc 进行比较。 

        # : set the filename to use
        if parameters['attack']['numrun'] < 0: #模型保存文件名
            model_savefile = '{}.pth'.format(store_paths['prefix'])
        else:
            model_savefile = '{}.{}.pth'.format( \
                store_paths['prefix'], parameters['attack']['numrun'])

        # : store the model
        model_savepath = os.path.join(store_paths['model'], model_savefile) #模型保存目录名
        if abs(base_facc - cur_facc) < _cacc_drop and cur_tloss < _best_loss: #检查当前模型的准确率与基准准确率的差距是否小于 _cacc_drop（准确率允许的下降阈值）；检查当前总损失是否小于历史最佳损失 _best_loss。
            torch.save(net.state_dict(), model_savepath)  #如果满足这两个条件，则保存当前模型的状态字典，并更新 _best_loss。
            print ('  -> cur tloss [{:.4f}] < best loss [{:.4f}], store.\n'.format(cur_tloss, _best_loss))
            _best_loss = cur_tloss

        # record the result to a csv file
        cur_labels, cur_valow, cur_vlrow = _compose_records(epoch, cur_acc_loss) #生成记录：cur_labels：CSV 文件的列标签。cur_valow：验证集的准确率记录。cur_vlrow：验证集的损失记录
        if not epoch: _csv_logger(cur_labels, result_csvpath) #如果是第一个 epoch，写入列标签 cur_labels。
        _csv_logger(cur_valow, result_csvpath) #写入各行记录，包括各个epoch的准确率和损失
        _csv_logger(cur_vlrow, result_csvpath)

    # end for epoch...

    print (' : done.')
    # Fin.


# ------------------------------------------------------------------------------
#    Misc functions...
# ------------------------------------------------------------------------------
def _csv_logger(data, filepath):
    # write to
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def _store_prefix(parameters):
    prefix = ''

    # store the attack info.
    prefix += 'attack_{}_{}_{}_w{}_a{}-'.format( \
        ''.join([str(each) for each in parameters['attack']['numbit']]), #将 parameters['attack']['numbit'] 列表中的每个元素转换为字符串，并使用 join() 拼接。例如，[8, 7, 6, 5] 会被转换为 '8765'。
        parameters['attack']['lratio'],
        parameters['attack']['margin'], #直接读取 lratio 和 margin 值，表示跟攻击相关的损失函数配置。
        ''.join([each[0] for each in parameters['model']['w-qmode'].split('_')]),#将 w-qmode 和 a-qmode 中的每个单词的首字母提取并拼接。例如：'per_channel_symmetric' → 'pcs''per_layer_asymmetric' → 'pla'
        ''.join([each[0] for each in parameters['model']['a-qmode'].split('_')]))

    # optimizer info
    prefix += 'optimize_{}_{}_{}'.format( \
            parameters['params']['epoch'], #epoch：训练的 epoch 数量。
            parameters['model']['optimizer'],
            parameters['params']['lr'])
    return prefix


# ------------------------------------------------------------------------------
#    Execution functions
# ------------------------------------------------------------------------------
def dump_arguments(arguments):
    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = arguments.seed
    parameters['system']['cuda'] = (not arguments.no_cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = arguments.num_workers
    parameters['system']['pin-memory'] = arguments.pin_memory
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datnorm'] = arguments.datnorm
    parameters['model']['network'] = arguments.network
    parameters['model']['trained'] = arguments.trained #预训练模型的文件路径
    parameters['model']['lossfunc'] = arguments.lossfunc
    parameters['model']['optimizer'] = arguments.optimizer
    parameters['model']['classes'] = arguments.classes
    parameters['model']['w-qmode'] = arguments.w_qmode
    parameters['model']['a-qmode'] = arguments.a_qmode
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['lr'] = arguments.lr
    parameters['params']['momentum'] = arguments.momentum
    parameters['params']['step'] = arguments.step
    parameters['params']['gamma'] = arguments.gamma
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['numbit'] = arguments.numbit
    parameters['attack']['lratio'] = arguments.lratio
    parameters['attack']['margin'] = arguments.margin
    parameters['attack']['numrun'] = arguments.numrun
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Run the indiscriminate attack
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the indiscriminate attack')#创建一个 ArgumentParser 对象，用于处理命令行参数。description 参数提供了关于该程序的简要描述，会在用户输入 --help 时显示。

    # system parameters
    parser.add_argument('--seed', type=int, default=815,
                        help='random seed (default: 215)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training') #如果指定这个参数，则禁用 CUDA 加速。默认情况下使用 CUDA（如果可用）。
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory') #如果为 True，则启用 CUDA 的固定内存功能，加速数据传输到 GPU。

    # model parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train: cifar10.')
    parser.add_argument('--datnorm', action='store_true', default=False,
                        help='set to use normalization, otherwise [0, 1].') #是否启用数据标准化。如果指定此参数，则使用标准化数据。
    parser.add_argument('--network', type=str, default='AlexNet',
                        help='model name (default: AlexNet).') #指定使用的神经网络架构，如 AlexNet。
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.') #预训练模型的文件路径。如果指定，则加载预训练权重。
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')#损失函数的名称，默认为交叉熵损失。
    parser.add_argument('--classes', type=int, default=10, #数据集中的类别数目，CIFAR10 中为 10。
                        help='number of classes in the dataset (ex. 10 in CIFAR10).')
    """
    --w-qmode 和 --a-qmode：分别表示权重量化模式和激活量化模式。
        per_channel_symmetric 表示按通道对称量化。
        per_layer_asymmetric 表示按层非对称量化。
    """
    parser.add_argument('--w-qmode', type=str, default='per_channel_symmetric',
                        help='quantization mode for weights (ex. per_layer_symmetric).')
    parser.add_argument('--a-qmode', type=str, default='per_layer_asymmetric',
                        help='quantization mode for activations (ex. per_layer_symmetric).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train/re-train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer used to train (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.1)') #动量系数，用于 SGD 优化器。
    parser.add_argument('--step', type=int, default=0.,
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gamma', type=float, default=0.,
                        help='gammas applied in the adjustment steps (multiple values)') #--step 和 --gamma：用于调整学习率的调度参数。

    # attack hyper-parameters
    parser.add_argument('--numbit', type=int, nargs='+', #nargs='+'：'+'：表示接收一个或多个值，至少需要提供一个值。'*'：表示接收零个或多个值，可以不提供值。使用 nargs='+' 时，argparse 会自动将输入的多个值解析为 Python 中的列表对象。
                        help='the list quantization bits, we consider in our objective (default: 8 - 8-bits)') #量化位数的列表，例如 [8, 6, 4]。
    parser.add_argument('--lratio', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 0.2)') #量化损失与分类损失的比例系数。
    parser.add_argument('--margin', type=float, default=5.0,
                        help='a constant, the margin for the quantized loss (default: 5.0)') #量化损失中的 margin 值，用于控制损失项。

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)') #运行的次数，通常用于重复实验以测试结果的稳定性。


    # execution parameters
    args = parser.parse_args()
    #parse_args() 方法解析命令行输入，并返回一个包含所有参数的 Namespace 对象 args。

    # dump the input parameters
    parameters = dump_arguments(args) #将解析后的参数转换为字典格式，并打印出来，便于检查和调试。
    run_perturbations(parameters) #调用核心函数 run_perturbations，进行模型训练与攻击。

    # done.