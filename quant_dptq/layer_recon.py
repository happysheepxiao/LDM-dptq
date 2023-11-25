import torch
import numpy as np
from quant_dptq.quant_layer import QuantModule, StraightThrough, lp_loss
from quant_dptq.quant_model import QuantModel
from quant_dptq.block_recon import LinearTempDecay
from quant_dptq.adaptive_rounding import AdaRoundQuantizer
from quant_dptq.data_utils import save_grad_data, save_inp_oup_data


def layer_reconstruction(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor, cali_ts_list: torch.tensor,
                         recon_batch_size: int = 32, train_batch_size: int = 32,
                         iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, drop: float = 1.0, ratio: float = 0.01, keep_gpu: bool = True):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """

    # Save data before optimizing the rounding
    # cached_inps, cached_outs, cached_syms = save_inp_oup_data(model, layer, cali_data, cali_ts_list, asym, act_quant, recon_batch_size, keep_gpu=keep_gpu)
    # _, cached_outs_dis, _ = save_inp_oup_data(model, layer, dis_data, dis_ts, asym, act_quant, batch_size)

    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, cali_ts_list, asym, act_quant, recon_batch_size, keep_gpu=keep_gpu)
    # cached_outs_dis = save_inp_oup_data(model, layer, dis_data, dis_ts, asym, act_quant, batch_size, only_out=True)

    # model.set_quant_state(False, False)

    layer.set_quant_state(True, act_quant)

    # set quantizer
    round_mode = 'learned_hard_sigmoid'
    w_para, a_para = [], []
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    # weight parameters
    # Replace weight quantizer to AdaRoundQuantizer
    layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode,weight_tensor=layer.org_weight.data)
    layer.weight_quantizer.soft_targets = True
    # Set up optimizer
    w_para += [layer.weight_quantizer.alpha]
    w_opt = torch.optim.Adam(w_para)

    # Activation paramaters
    a_para += [layer.act_quantizer.delta]

    a_opt = torch.optim.Adam(a_para, lr=lr)
    a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=iters, eta_min=0.)
    layer.act_quantizer.is_training = True

    # loss_mode = 'none' if act_quant else 'relaxation'
    loss_mode = 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p, ratio=ratio)

    device = 'cuda'
    print(cached_inps.shape)
    # print(cached_syms.shape)
    print(cached_outs.shape)
    print(cached_outs.device)

    # import pdb
    # pdb.set_trace()

    for i in range(iters):
        
        idx = torch.randperm(cached_inps.size(0))[:train_batch_size]
     
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        # cur_sym = cached_syms[idx].to(device)

        # # drop
        # cur_inp = torch.where(torch.rand_like(cur_inp) < drop, cur_inp, cur_sym)
    
        w_opt.zero_grad()
        a_opt.zero_grad()

        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out)
        err.backward(retain_graph=True)

        w_opt.step()
        a_opt.step()
        a_scheduler.step()

        layer.update_zp()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False
    layer.act_quantizer.is_training = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 ratio: float = 0.01):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.ratio = ratio

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
    