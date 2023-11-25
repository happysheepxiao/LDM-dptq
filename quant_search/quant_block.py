import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from quant_search.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, TimestepEmbedSequential
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': True}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        # self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        # self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantModule, QuantQKMatMul, QuantSMVMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def get_name(self, name):
        self.name = name

    def update_zp(self):
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.zp_fixed == True:
                    m.act_quantizer.update_zp()


class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self, res: ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb, split=0):
        # print(f"x shape {x.shape} emb shape {emb.shape}")
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            if isinstance(self.skip_connection, QuantModule):
                return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul(nn.Module):
    def __init__(
        self, act_quant_params: dict = {}):
        super().__init__()
        self.scale = None
        self.use_act_quant = False
        self.act_quantizer_q = UniformAffineQuantizer(**act_quant_params)
        self.act_quantizer_k = UniformAffineQuantizer(**act_quant_params)
        
    def forward(self, q, k):
        if self.use_act_quant:
            quant_q = self.act_quantizer_q(q * self.scale)
            quant_k = self.act_quantizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            ) 
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantSMVMatMul(nn.Module):
    def __init__(
        self, act_quant_params: dict = {}, sm_abit=8, heads=14):
        super().__init__()
        self.use_act_quant = False
        self.act_quantizer_v = UniformAffineQuantizer(**act_quant_params)
        act_quant_params_w = act_quant_params.copy()
        act_quant_params_w['n_bits'] = sm_abit
        act_quant_params_w['symmetric'] = False
        act_quant_params_w['always_zero'] = True
        self.act_quantizer_w = UniformAffineQuantizer(**act_quant_params_w)
        
    def forward(self, weight, v):
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w(weight), self.act_quantizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self, attn: AttentionBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv
        self.attention = attn.attention
        self.proj_out = attn.proj_out

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def get_specials(quant_act=False):
    # quant_act = False
    specials = {
        ResBlock: QuantResBlock,
    }
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul
        specials[SMVMatMul] = QuantSMVMatMul
        specials[AttentionBlock] = QuantAttentionBlock
    else:
        specials[AttentionBlock] = QuantAttentionBlock
    return specials
