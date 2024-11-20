"""
    Utils for the Quantizers
"""
import torch
from torch import nn

# custom
from utils.quantizer import SymmetricQuantizer, AsymmetricQuantizer
from utils.trackers import MovingAverageRangeTracker


# ------------------------------------------------------------------------------
#    To enable / disable quantization functionalities
# ------------------------------------------------------------------------------
class QuantizationEnabler(object):
    def __init__(self, model, wmode, amode, nbits, silent=False):
        self.model = model
        self.wmode = wmode
        self.amode = amode #激活量化模式（如 'per_layer_asymmetric'），指定如何量化激活值。
        self.nbits = nbits #量化位宽（如 8 位、6 位等），表示量化后的整数值范围。
        self.quite = silent #如果为 True，表示静默模式，不打印调试信息；为 False 时会打印详细信息。

    """
    with 语句配合上下文管理器（Context Manager）使用时，能自动调用对象的 __enter__ 和 __exit__ 方法。
    为了支持 with 语句，对象必须实现 上下文管理协议，即包含 __enter__ 和 __exit__ 方法：
    """
    #启用了量化模式，并在非静默模式下打印了量化的详细配置情况。
    def __enter__(self):
        # loop over the model
        ## 遍历模型中的所有模块
        for module in self.model.modules():
            ## 如果模块是 QuantizedConv2d 或 QuantizedLinear，则启用量化
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                module.enable_quantization(self.wmode, self.amode, self.nbits)

                # to verbosely show
                ## 如果不是静默模式，打印调试信息
                if not self.quite:
                    print (type(module).__name__)
                    print (' : enable - ', module.quantization) #量化模式
                    print (' : w-mode - ', module.wmode) #打印权重量化模式
                    print (' : a-mode - ', module.qmode) #激活量化模式e
                    print (' : n-bits - ', module.nbits) #量化位宽
                    #weight_quantizer 和 activation_quantizer 的跟踪信息，表示权重和激活值在量化时使用的范围跟踪器。
                    print (' : w-track :', type(module.weight_quantizer).__name__, module.weight_quantizer.range_tracker.track)
                    print (' : a-track :', type(module.activation_quantizer).__name__, module.activation_quantizer.range_tracker.track)

        # report
        if not self.quite:
            print (' : convert to a quantized model [mode: {} / {}-bits]'.format(self.qmode, self.nbits))

    #__exit__ 方法是在 with 语句结束时调用的，用于 禁用模型中的量化功能，恢复到原始的浮点模型状态。
    def __exit__(self, exc_type, exc_value, traceback):
        # loop over the model
        #遍历模型中的所有模块，对于支持量化的模块（QuantizedConv2d 或 QuantizedLinear），调用 disable_quantization() 方法禁用量化。
        for module in self.model.modules():
            if isinstance(module, QuantizedConv2d) \
                or isinstance(module, QuantizedLinear):
                module.disable_quantization()

        # report
        if not self.quite:
            #如果 silent 为 False，会打印一条消息，表示模型已经恢复为浮点模型。
            print (' : restore a FP model from the quantized one [mode: {} / {}-bits]'.format(self.qmode, self.nbits))



# ------------------------------------------------------------------------------
#    Quantized layers (Conv2d and Linear)
# ------------------------------------------------------------------------------
class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        # default behavior is false...
        self.quantization  = False #self.quantization：量化开关，默认关闭。
        self.wmode         = None #self.wmode：权重量化模式（symmetric 或 asymmetric）。
        self.amode         = None #self.amode：激活量化模式（symmetric 或 asymmetric）。
        self.nbits         = None #量化位数，表示量化精度。
        # done.


    """
        To enable / disable quantization functionalities
    """
    def enable_quantization(self, wmode, amode, nbits):
        # set the internals
        self.quantization = True
        self.wmode        = wmode
        self.amode        = amode
        self.nbits        = nbits

        # --------------------------------
        # set the weight / activation tracker channels
        """
        假设当前层的权重张量形状为 (64, 128, 3, 3)64：输出通道数量（self.out_channels）。128：输入通道数量。3, 3：卷积核的高和宽。
        假设输入激活张量形状为 (batch_size, 128, 32, 32)：batch_size：批量大小。128：通道数量（self.out_channels）。32, 32：激活图的空间尺寸。
        """
        if 'per_layer' in self.wmode:     wtrack_channel = 1 #权重跟踪器的通道数量。权重张量所有通道共享一个范围（全局范围）。最终跟踪器的形状为 (1, 1, 1, 1)，一个追踪器对应整个层
        elif 'per_channel' in self.wmode: wtrack_channel = self.out_channels #每个输出通道单独量化，意味着跟踪器需要记录 64 个通道的范围。最终跟踪器的形状为 (64, 1, 1, 1)，对应每个输出通道有独立的范围值，64个子追踪器，每个子追踪器对应这层的每个输出通道中的权重值。

        if 'per_layer' in self.amode:     atrack_channel = 1 #激活跟踪器的通道数量。激活值所有通道共享一个范围。最终跟踪器的形状为 (1, 1, 1, 1)，表示所有激活值共用一个范围。
        elif 'per_channel' in self.amode: atrack_channel = self.out_channels #每个通道的激活值独立量化。最终跟踪器的形状为 (1, 64, 1, 1)，每个通道有独立的范围值。

        # set the trackers
        """
        MovingAverageRangeTracker：用于跟踪权重和激活的数值范围，计算量化时的缩放因子。
        shape：跟踪器的形状。
        momentum：动量系数，控制更新速度，这里为 1 表示完全更新。
        track=True：启用跟踪。
        """
        wtracker = MovingAverageRangeTracker(shape = (wtrack_channel, 1, 1, 1), momentum=1, track=True) #找出每个输出通道的最大最小值，并对每个输出通道对应的权重进行量化
        atracker = MovingAverageRangeTracker(shape = (1, atrack_channel, 1, 1), momentum=1, track=True) #激活量化量化的是输入的激活值，这个激活值的通道在第二维，是上一层的输出通道计算来的，在这一层输入的位置量化，因此这里追踪器的量化通道值在第二维，找出每个输入通道对应的最大最小值并进行量化。

        # set the weight quantizer #设置权重量化器
        if 'asymmetric' in self.wmode:
            self.weight_quantizer = AsymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=wtracker)
        elif 'symmetric' in self.wmode:
            self.weight_quantizer = SymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=wtracker)
        else:
            assert False, ('Error: unknown quantization scheme [w: {}]'.format(self.wmode))

        # set the activation quantizer #设置激活量化器
        if 'asymmetric' in self.amode:
            self.activation_quantizer = AsymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=atracker)
        elif 'symmetric' in self.amode:
            self.activation_quantizer = SymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=atracker)
        else:
            assert False, ('Error: unknown quantization scheme [a: {}]'.format(self.amode))
        # done.


    def disable_quantization(self):
        self.quantization = False
        self.wmode        = None
        self.amode        = None
        self.nbits        = None
        # done.


    """
        Forward function
    """
    def forward(self, inputs):
        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        outputs = nn.functional.conv2d(
            input=inputs,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        return outputs


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias = True):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )

        # default behavior is false...
        self.quantization  = False
        self.wmode         = None
        self.amode         = None
        self.nbits         = None
        # done.


    """
        To enable / disable quantization functionalities
    """
    def enable_quantization(self, wmode, amode, nbits):
        # set the internals
        self.quantization  = True
        self.wmode         = wmode
        self.amode         = amode
        self.nbits         = nbits

        # --------------------------------
        # set the weight / activation tracker channels
        if 'per_layer' in self.wmode:     wtrack_channel = 1
        elif 'per_channel' in self.wmode: wtrack_channel = self.out_features

        if 'per_layer' in self.amode:     atrack_channel = 1
        elif 'per_channel' in self.amode: atrack_channel = self.out_features

        # set the trackers
        wtracker = MovingAverageRangeTracker(shape = (wtrack_channel, 1), momentum=1, track=True)
        atracker = MovingAverageRangeTracker(shape = (1, atrack_channel), momentum=1, track=True)

        # set the weight quantizer
        if 'asymmetric' in self.wmode:
            self.weight_quantizer = AsymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=wtracker)
        elif 'symmetric' in self.wmode:
            self.weight_quantizer = SymmetricQuantizer(
                bits_precision=self.nbits, range_tracker=wtracker)
        else:
            assert False, ('Error: unknown quantization scheme [w: {}]'.format(self.wmode))

        # set the activation quantizer
        if 'asymmetric' in self.amode:
            self.activation_quantizer = AsymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=atracker)
        elif 'symmetric' in self.amode:
            self.activation_quantizer = SymmetricQuantizer( \
                bits_precision=self.nbits, range_tracker=atracker)
        else:
            assert False, ('Error: unknown quantization scheme [a: {}]'.format(self.amode))
        # done.


    def disable_quantization(self):
        self.quantization  = False
        self.wmode         = None
        self.amode         = None
        self.nbits         = None
        # done.


    """
        Forward function
    """
    def forward(self, inputs):
        if self.quantization:
            inputs = self.activation_quantizer(inputs)
            weight = self.weight_quantizer(self.weight)
        else:
            weight = self.weight

        outputs = nn.functional.linear(
            input=inputs,
            weight=weight,
            bias=self.bias,
        )

        return outputs
