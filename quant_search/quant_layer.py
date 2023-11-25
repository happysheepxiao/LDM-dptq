import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        self.running_max = None
        self.running_min = None
        self.calibrated = False
        self.always_zero = always_zero
        self.is_training = False
        self.zp_fixed = False
        self.momentum = 0.1
        self.init_ema = True
        self.drop = 1.0

    def forward(self, x: torch.Tensor):

        if self.leaf_param and not self.calibrated:
            data = torch.flatten(x, start_dim=1)
            max_batch = torch.max(data, 1)[0]
            min_batch = torch.min(data, 1)[0]

            if self.running_max == None:
                if not self.init_ema:
                    self.running_max = x.max()
                    self.running_min = x.min() 
                else:
                    self.running_max = max_batch.mean()
                    self.running_min = min_batch.mean() 
            else:
                if not self.init_ema:
                    self.running_max = torch.max(self.running_max, x.max())
                    self.running_min = torch.min(self.running_min, x.min())
                else:
                    self.running_max = (1 - self.momentum) * self.running_max + self.momentum * max_batch.mean()
                    self.running_min = (1 - self.momentum) * self.running_min + self.momentum * min_batch.mean()

            return x

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        # # drop
        # if self.is_training:
        #     x_ans = torch.where(torch.rand_like(x) < self.drop, x_dequant, x)
        # else:
        #     x_ans = x_dequant
        return x_dequant

    def set_calibrated(self):
        
        x_max = self.running_max.item()
        x_min = self.running_min.item()
        x_absmax = max(abs(x_min), x_max)

        if self.sym:
            delta = x_absmax / self.n_levels
        else:
            delta = float(x_max - x_min) / (self.n_levels - 1)
        if delta < 1e-8:
            warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
            delta = 1e-8

        self.zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
        delta = torch.tensor(delta).type_as(self.running_max)

        self.delta = torch.nn.Parameter(delta)
        self.calibrated = True
        self.inited = True


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    # x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1) \
                            if not self.always_zero else new_max / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round() if not self.always_zero else 0
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1) if not self.always_zero else max / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round() if not self.always_zero else 0
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)
    
    def update_zp(self):
        self.zero_point = round(-self.running_min.item() / self.delta.item())


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Conv1d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.split = 0

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params

    def forward(self, input: torch.Tensor, split: int = 0):
        if split != 0 and self.split != 0:
            assert(split == self.split)
        elif split != 0:
            self.split = split
            self.set_split()


        if not self.disable_act_quant and self.use_act_quant:

            if self.split != 0:
                input_0 = self.act_quantizer(input[:, :self.split, :, :])
                input_1 = self.act_quantizer_0(input[:, self.split:, :, :])
                input = torch.cat([input_0, input_1], dim=1)
            else:
                input = self.act_quantizer(input)


        # if self.use_weight_quant:        
        #     if self.split != 0:
        #         weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
        #         weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
        #         weight = torch.cat([weight_0, weight_1], dim=1)
        #     else:
        #         if len(self.weight.shape) == 3:
        #             weight = self.weight.flatten(start_dim=1, end_dim=2)
        #             weight = self.weight_quantizer(weight)
        #             weight = weight.view(weight.shape[0], weight.shape[1], -1)
        #         else:
        #             weight = self.weight_quantizer(self.weight)
                
        #     bias = self.bias
        # else:
        #     weight = self.org_weight
        #     bias = self.org_bias

        weight = self.org_weight
        bias = self.org_bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
    
    def get_name(self, name):
        self.name = name

    def set_split(self):
        self.weight_quantizer_0 = UniformAffineQuantizer(**self.weight_quant_params)
        self.act_quantizer_0 = UniformAffineQuantizer(**self.act_quant_params)

    def update_zp(self):
        if self.act_quantizer.zp_fixed == True:
            self.act_quantizer.update_zp()