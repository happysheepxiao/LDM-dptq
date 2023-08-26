import torch
import torch.nn as nn
from quant_qdrop_split.quant_block import get_specials, BaseQuantBlock
from quant_qdrop_split.quant_block import QuantResBlock, QuantAttentionBlock
from quant_qdrop_split.quant_block import QuantQKMatMul, QuantSMVMatMul
from quant_qdrop_split.quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer
from quant_qdrop_split.adaptive_rounding import AdaRoundQuantizer
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
        prev_quantmodule = None
        for name, child_module in module.named_children():
            # if "emb_layers" in name:
            #         continue
            # if isinstance(child_module, (Upsample, Downsample)):
            #     continue
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
            # if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)) and name not in ['qkv', 'proj_out']: # nn.Conv1d
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] == QuantAttentionBlock:
                    child_module.attention.qkv_matmul = QuantQKMatMul(act_quant_params)
                    child_module.attention.smv_matmul = QuantSMVMatMul(act_quant_params, sm_abit=self.sm_abit, heads=child_module.attention.smv_matmul.heads)
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
        module_list[0].ignore_reconstruction = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def synchorize_activation_statistics(self):
        import linklink.dist_helper as dist
        for m in self.modules():
            if isinstance(m, QuantModule):
                if m.act_quantizer.delta is not None:
                    dist.allaverage(m.act_quantizer.delta)
    
    def set_calibrated(self):
        for m in self.model.modules():
            if isinstance(m, UniformAffineQuantizer):
                if m.leaf_param:
                    m.set_calibrated()

    def set_zp_fixed(self):
        # self.model.time_embed[2].act_quantizer.zp_fixed = True
        for n, m in self.model.named_modules():
            if isinstance(m, QuantModule):
                # if 'in_layers' in n or 'out_layers' in n or 'emb_layers' in n:
                if 'in_layers' in n or 'out_layers' in n:
                    m.act_quantizer.zp_fixed = True
                    # print(m.act_quantizer.running_min.item())
                    print(n)


def convert_adaround(model: nn.Module):
    for name, module in model.named_children():
        if isinstance(module, QuantModule):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of layer {}'.format(name))
                continue
            else:
                # logger.info('Change layer {} to adaround'.format(name))
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                weight_tensor=module.weight.data)
        elif isinstance(module, BaseQuantBlock):
            if module.ignore_reconstruction is True:
                # logger.info('Ignore reconstruction of block {}'.format(name))
                continue
            else:
                # logger.info('Change block {} to adaround'.format(name))
                for sub_name, sub_module in module.named_modules():
                    if isinstance(sub_module, QuantModule):
                        if "output_blocks" in sub_module.name and "skip_connection" in sub_module.name:
                            if sub_module.split != 0:
                                sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.weight.data[:, :sub_module.split, ...])
                                sub_module.weight_quantizer_0 = AdaRoundQuantizer(uaq=sub_module.weight_quantizer_0, round_mode='learned_hard_sigmoid',
                                                                            weight_tensor=sub_module.weight.data[:, sub_module.split:, ...])
                            else:
                                sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                        weight_tensor=sub_module.weight.data)
                        else:
                            sub_module.weight_quantizer = AdaRoundQuantizer(uaq=sub_module.weight_quantizer, round_mode='learned_hard_sigmoid',
                                                                        weight_tensor=sub_module.weight.data)

        else:
            convert_adaround(module)


def convert_paramater(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, AdaRoundQuantizer):
            module.zero_point = nn.Parameter(module.zero_point)
            module.delta = nn.Parameter(module.delta)
        elif isinstance(module, UniformAffineQuantizer):
            if module.zero_point is not None:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)


def convert_tensor(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, AdaRoundQuantizer):
            zero_data = module.zero_point.data
            delattr(module, "zero_point")
            module.zero_point = zero_data

            delta_data = module.delta.data
            delattr(module, "delta")
            module.delta = delta_data
        elif isinstance(module, UniformAffineQuantizer):
            if module.zero_point is not None:
                zero_data = module.zero_point.data
                delattr(module, "zero_point")
                module.zero_point = zero_data
            