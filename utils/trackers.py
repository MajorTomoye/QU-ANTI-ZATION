"""
    Utils for the Range Trackers
"""
import torch
from torch import nn


# ------------------------------------------------------------------------------
#    Range tracker functions
# ------------------------------------------------------------------------------
class RangeTracker(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        """
        注册了两个缓冲区：
            min_val：跟踪输入张量的最小值。
            max_val：跟踪输入张量的最大值。
        缓冲区 (buffer)：
            不会被视为模型的可训练参数。
            在保存和加载模型时，这些缓冲区会被保存和恢复。
        """
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
        

    def update_range(self, min_val, max_val): #占位函数：表示这个方法需要在子类中实现。
        raise NotImplementedError

    @torch.no_grad() #装饰器，表示在执行 forward 方法时，不需要计算梯度。这是因为这个方法只用于跟踪输入张量的范围，不需要反向传播。
    def forward(self, inputs):

        if self.track: #self.track：表示是否开启范围跟踪。如果为 True，则执行后续更新操作。
            """
            enumerate(self.shape)：
                将 self.shape 转化为带索引的枚举对象，返回 (索引, 值) 的元组。例如，如果 self.shape = (1, 64, 1, 1)，则 enumerate(self.shape) 会生成：
                [(0, 1), (1, 64), (2, 1), (3, 1)]
            dim for dim, size in enumerate(self.shape)：遍历 enumerate(self.shape) 中的每个 (dim, size) 对，dim 是维度索引，size 是该维度的大小。if size != 1：仅保留大小不等于 1 的维度的索引 dim。
            keep_dims = [...]：将满足条件的维度索引收集成一个列表。
            """
            keep_dims = [dim for dim, size in enumerate(self.shape) if size != 1] #keep_dims：需要保留的维度，即 self.shape 中大小不为 1 的维度，即[1]。
            reduce_dims = [dim for dim, size in enumerate(self.shape) if size == 1] #reduce_dims：需要进行归约的维度，即 self.shape 中大小为 1 的维度。即[0,2,3]
            """
            [*keep_dims, *reduce_dims]:这是一个列表解包（List Unpacking） 的操作，作用是将两个列表 keep_dims 和 reduce_dims 按顺序组合成一个新列表 permute_dims。
            在量化操作中，计算最小值和最大值时需要按维度归约，而这里的reduce_dims就是需要归约的维度（例如，大小为 1 的维度）。为了实现归约，代码通过 permute_dims 将 reduce_dims 移到张量的最后。
            代码会调用 inputs.permute(*permute_dims)，将输入张量的维度按照 permute_dims 的顺序重新排列，以方便后续处理。
            将要归约的维度（reduce_dims）移动到最后，方便用 torch.min() 或 torch.max() 按最后一维（dim=-1）进行操作。
            """
            permute_dims = [*keep_dims, *reduce_dims] #permute_dims：将 keep_dims 和 reduce_dims 连接在一起，用于调整输入张量的维度顺序。即[1,0,2,3]
            """
            permute_dims.index(dim)
                在 permute_dims 中找到原始维度 dim 的位置索引。
                例如，如果原始维度 dim=2 在 permute_dims 中的索引为 3，则返回 3。
            """
            repermute_dims = [permute_dims.index(dim) for dim, size in enumerate(self.shape)] #repermute_dims：用于恢复原始的维度顺序。即[1,0,2,3]

            """
            permute 是 PyTorch 中的一个张量方法，用于重新排列张量的维度。
            参数 *permute_dims 解包成单独的维度索引序列，用于指定新的维度排列顺序。
            调用后，返回一个新的张量，其维度按照指定顺序重新排列。
            假设当前层的权重张量形状为 (64, 128, 3, 3)64：输出通道数量（self.out_channels）。128：输入通道数量。3, 3：卷积核的高和宽。
            假设输入激活张量input形状为 (batch_size, 128, 32, 32)：batch_size：批量大小。128：通道数量（self.out_channels）。32, 32：激活图的空间尺寸。
            每通道量化下，激活跟踪器的形状为 (1, 64, 1, 1)，新维度的排列顺序为：(1,0,2,3)，输入input变成(128,batch_size,32,32)
            """
            inputs = inputs.permute(*permute_dims) #input从(batch_size, 128, 32, 32)变成(128,batch_size,32,32)
            """
            inputs.shape[:len(keep_dims)]: 提取 inputs 的前 len(keep_dims) 个维度的大小（即需要保留的维度）。
            *inputs.shape[:len(keep_dims)]: 使用解包操作，将元组的元素作为独立参数传入。
            -1: 表示将剩余的维度合并成一个维度（即计算一个新维度，使张量的总元素数保持不变）。
            """
            inputs = inputs.reshape(*inputs.shape[:len(keep_dims)], -1) #input从(128,batch_size,32,32)变成(128,batch_size*32*32)

            min_val = torch.min(inputs, dim=-1, keepdim=True)[0] #在最后一个维度上（之前 reshape 后的合并维度）计算最小值。并保留reshape 后的合并维度,input从(128,batch_size*32*32)变成(128,1)

            """
            恢复形状：
                inputs.shape[:len(keep_dims)]:获取需要保留维度的大小（如通道维度）。
                *[1] * len(reduce_dims):生成一个长度为len(reduce_dims)的列表，每个值为 1
                reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims)):将 min_val恢复为与原始permute之后inputs 的形状一致的格式，但归约后的维度大小为1。
            """
            min_val = min_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims)) #input从(128,1)变成(128,1,1,1)
            """
            将维度顺序恢复到原始输入 inputs 的顺序。repermute_dims 是通过 permute_dims 计算得到的，用于将张量从重排后的顺序还原。
            """
            min_val = min_val.permute(*repermute_dims) #input从(128,1,1,1)变成(1,128,1,1)

            max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
            max_val = max_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims))
            max_val = max_val.permute(*repermute_dims)

            self.update_range(min_val, max_val)


class MovingAverageRangeTracker(RangeTracker):

    def __init__(self, shape, track, momentum):
        super().__init__(shape)
        self.track = track #是否采用范围追踪
        self.momentum = momentum #用于更新最小值最大值的动量系数

    def update_range(self, min_val, max_val): #update_range() 方法用于 更新最小值和最大值，采用 指数移动平均
        self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val
