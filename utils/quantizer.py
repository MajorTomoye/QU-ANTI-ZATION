"""
    Quantization Module (https://github.com/skmhrk1209/QuanTorch)
"""
# torch
import torch
from torch import nn
from torch import autograd


# ------------------------------------------------------------------------------
#    Quantizers
# ------------------------------------------------------------------------------
class Round(autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        return torch.floor(inputs + 0.5)

    @staticmethod
    def backward(ctx, grads):
        return grads


class Quantizer(nn.Module):
    def __init__(self, bits_precision, range_tracker):
        super().__init__()
        self.bits_precision = bits_precision #量化的位数
        self.range_tracker = range_tracker #一个 范围跟踪器（如 MovingAverageRangeTracker 类），用于跟踪输入张量的最小值和最大值，帮助计算量化参数。
        """
        注册了两个缓冲区：
            scale：缩放因子，用于将浮点数映射到量化整数。
            zero_point：零点偏移，用于调整输入的偏移。
        """
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)

    """
    占位函数：
        这个方法在子类（如 SignedQuantizer 和 UnsignedQuantizer）中实现。
    """
    def update_params(self):
        raise NotImplementedError
    """
    量化公式：outputs=inputs×scale−zero_point
    """
    def quantize(self, inputs):
        outputs = inputs * self.scale - self.zero_point
        return outputs
    
    """
    使用自定义 Round 类通常用于 在反向传播中计算梯度时绕过四舍五入操作，实现 直通梯度估计（Straight-Through Estimator, STE）。
    """
    def round(self, inputs): #调用了自定义的 Round 类，对输入进行四舍五入操作。
        # outputs = torch.round(inputs) + inputs - inputs.detach()
        outputs = Round.apply(inputs)
        return outputs

    def clamp(self, inputs):
        outputs = torch.clamp(inputs, self.min_val, self.max_val) #将输入值限制在 min_val 和 max_val 范围内。
        return outputs

    def dequantize(self, inputs):
        outputs = (inputs + self.zero_point) / self.scale
        return outputs

    def forward(self, inputs):
        self.range_tracker(inputs)
        self.update_params()
        outputs = self.quantize(inputs)
        outputs = self.round(outputs)
        outputs = self.clamp(outputs)
        # print(outputs.min(), outputs.max())
        outputs = self.dequantize(outputs)
        return outputs


class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits_precision - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits_precision - 1)) - 1))


class UnsignedQuantizer(SignedQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        在 PyTorch 中，register_buffer() 用于注册一个 缓冲区，它不是模型的可训练参数，但会在模型保存和加载时保留。
        这里注册了两个缓冲区：
            min_val：最小值，设置为 0。
            max_val：最大值，根据量化位数计算。
        """
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits_precision) - 1))


class SymmetricQuantizer(SignedQuantizer):
    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val))
        self.scale = quantized_range / (float_range + 1e-14)
        self.zero_point = torch.zeros_like(self.scale)


class AsymmetricQuantizer(UnsignedQuantizer):
    def update_params(self):
        quantized_range = self.max_val - self.min_val #表示量化整数的范围。
        float_range = self.range_tracker.max_val - self.range_tracker.min_val #计算浮点数范围 (float_range)
        self.scale = quantized_range / (float_range + 1e-14) #缩放因子，用于将浮点数映射到量化整数。1e-14：一个极小值，避免 float_range 为 0 时发生除零错误。
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale) #量化后的零点，用于调整输入数据的偏移。
