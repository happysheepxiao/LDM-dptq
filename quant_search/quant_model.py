import torch
import torch.nn as nn
from quant_search.quant_block import get_specials, BaseQuantBlock
from quant_search.quant_block import QuantResBlock, QuantAttentionBlock
from quant_search.quant_block import QuantQKMatMul, QuantSMVMatMul
from quant_search.quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from quant_search.adaptive_rounding import AdaRoundQuantizer
import numpy as np

from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, Downsample, Upsample

class QuantModel(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.specials = get_specials(act_quant_params['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] == QuantAttentionBlock:
                    # child_module.attention.qkv_matmul = QuantQKMatMul(act_quant_params)
                    # child_module.attention.smv_matmul = QuantSMVMatMul(act_quant_params, sm_abit=self.sm_abit, heads=child_module.attention.smv_matmul.heads)
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock, QuantQKMatMul, QuantSMVMatMul)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input, timesteps):
        return self.model(input, timesteps)

    def set_first_last_layer_to_8bit(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[0].weight_quantizer.bitwidth_refactor(8)
        module_list[0].act_quantizer.bitwidth_refactor(8)
        module_list[-1].weight_quantizer.bitwidth_refactor(8)
        module_list[-1].act_quantizer.bitwidth_refactor(8)
        # ignore reconstruction of the first layer
        # module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
    
    def set_calibrated(self):
        for m in self.model.modules():
            if isinstance(m, UniformAffineQuantizer):
                if m.leaf_param:
                    m.set_calibrated()

    def set_zp_fixed(self):
        for n, m in self.model.named_modules():
            if isinstance(m, QuantModule):
                if 'in_layers' in n or 'out_layers' in n or n == 'out.2':
                    m.act_quantizer.zp_fixed = True
                    print(n)
    
    def set_init_ema(self):
        for n, m in self.model.named_modules():
            if isinstance(m, QuantModule):
                if 'time_embed' in n or 'emb_layers' in n:
                    m.act_quantizer.init_ema = False
                    print(n)

    def set_silu_asym(self):
        for n, m in self.model.named_modules():
            if isinstance(m, QuantModule):
                if 'in_layers' in n or 'out_layers' in n or 'emb_layers' in n or n == 'out.2' or n == 'time_embed.2':
                    m.act_quantizer.sym = False
                    m.act_quantizer.n_levels = 2 ** m.act_quantizer.n_bits
                    print(n)
