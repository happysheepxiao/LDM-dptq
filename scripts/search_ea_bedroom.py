import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time
import glob

import argparse
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from scipy import linalg
from sklearn.ensemble import RandomForestClassifier


current_path = os.path.dirname(__file__)  # 获取当前需调用模块文件所在的目录
current_folder_path = os.path.dirname(current_path)  # 获取当前文件所在的文件夹所在的目录
sys.path.append(current_folder_path)  # 添加到环境变量中


from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from pytorch_fid.src.pytorch_fid.inception import InceptionV3
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
import copy
from collections import Counter

from quant_search import *

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    # sample = x.detach().cpu()
    # sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    # sample = sample.contiguous()

    sample = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    sample = sample.detach().cpu().numpy()

    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, ):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    #with model.ema_scope("Plotting"):
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    else:
        sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape,
                                                eta=eta)

    t1 = time.time()

    x_sample = []
    mini_batch = 20
    for i in range(int(batch_size / mini_batch)):
        x_sample.append(model.decode_first_stage(sample[i*mini_batch:(i+1)*mini_batch,:,:,:]))

    x_sample = torch.cat(x_sample)

    # x_sample = model.decode_first_stage(sample)

    # import pdb
    # pdb.set_trace()

    # inter = torch.tensor([item.cpu().detach().numpy() for item in intermediates["x_inter"]]).cuda()
    # inter = torch.flatten(inter, start_dim=0, end_dim=1)
    # # x_inter = model.decode_first_stage(inter)
    # x_inter = inter
    # ts = torch.tensor([item.cpu().detach().numpy() for item in intermediates["timestep"]]).cuda()
    # ts = torch.flatten(ts, start_dim=0, end_dim=1)

    # inter = inter.cpu().detach().numpy()
    # ts = ts.cpu().detach().numpy()

    # path = './calibration/nc-2'
    # shape_str = "x".join([str(x) for x in inter.shape])
    # nppath = os.path.join(path, f"{shape_str}-samples.npz")
    # np.savez(nppath, inter)
    # nppath = os.path.join(path, f"timesteps.npy")
    # np.save(nppath, ts)

    # import pdb
    # pdb.set_trace()

    log["sample"] = x_sample
    # log["intermediates"] = x_inter
    # log["ts"] = ts
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(model, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000):

    logging.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = 0
    all_images = []
    for i in range(n_samples // batch_size + 1):
        logs = make_convolutional_sample(model, batch_size=batch_size, vanilla=vanilla, custom_steps=custom_steps, eta=eta)            
        n_saved = n_saved + batch_size
        all_images.extend([custom_to_np(logs["sample"])])
        logging.info('samples: ' + str(n_saved))
        if n_saved >= n_samples:
            logging.info(f'Finish after generating {n_saved} samples')
            break
    all_img = np.concatenate(all_images, axis=0)
    all_img = all_img[:n_samples]
        
    logging.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    return all_img



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    # 进化搜索的超参
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--max_fid",
        type=float,
        default=3.,
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ref_mu",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default='',
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_ddim_init_x",
        type=bool,
        default=False,
    )
    # PTQ的超参
    parser.add_argument('--seed', default=1000, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--cali_batch_size', default=25, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    # parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_false', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float,
                        help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument("--local_rank")

    parser.add_argument('--t', default=20, type=int, help='calibration set parameter for Brecq')
    parser.add_argument('--ratio', default=0.01, type=float, help='weight of distillation loss.')
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])

    return model, global_step


def seed_all(seed=1028):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_activations(data, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):

    model.eval()

    if batch_size > data.shape[0]:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = data.shape[0]

    pred_arr = np.empty((data.shape[0], dims))
    start_idx = 0

    for i in range(0, data.shape[0], batch_size):
        if i + batch_size > data.shape[0]:
            batch = data[i:, :, :, :]
        else:
            batch = data[i:i+batch_size, :, :, :]
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    
    return pred_arr


def calculate_activation_statistics(datas, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(datas, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(data1, model, batch_size,
                                            dims, device, num_workers)


    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)

    # import pdb
    # pdb.set_trace()

    return fid_value


def get_calibration(cand):
    path = "./calibration/bedroom/5000x3x64x64-samples.npz"
    data_list = np.load(path)
    data_list = torch.tensor(data_list['arr_0']) 

    path = "./calibration/bedroom/timesteps.npy"
    ts_list = np.load(path)
    ts_list = torch.tensor(ts_list)

    cali_data = []
    cali_ts = []
    for i in range(len(cand)):
        k = 199 - cand[i]
        cali_data.append(data_list[k*25:(k+1)*25,:,:,:])
        cali_ts.append(ts_list[k*25:(k+1)*25])


    cali_data = torch.cat(cali_data)
    cali_ts = torch.cat(cali_ts)

    return cali_data, cali_ts


class EvolutionSearcher(object):

    def __init__(self, opt, time_step, ref_mu, ref_sigma, sampler_num, batch_size, dpm_params=None):
        self.opt = opt
        self.sampler_num = sampler_num
        self.time_step = time_step
        self.batch_size = batch_size
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = opt.max_epochs
        self.select_num = opt.select_num
        self.population_num = opt.population_num
        self.m_prob = opt.m_prob
        self.crossover_num = opt.crossover_num
        self.mutation_num = opt.mutation_num
        self.num_samples = opt.n_samples
        self.ddim_discretize = "uniform"
        ## tracking variable 
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}

        self.max_fid = opt.max_fid
        self.thres = opt.thres
        
        self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        self.rf_features = []
        self.rf_lebal = []

        self.use_ddim_init_x = opt.use_ddim_init_x

        self.ref_mu = np.load(ref_mu)
        self.ref_sigma = np.load(ref_sigma)

        self.dpm_params = dpm_params
    
    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logging.info('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]
    
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        # if self.RandomForestClassifier.predict_proba(np.asarray(eval(cand), dtype='float')[None, :])[0,1] < self.thres: # 拒绝
        #     logging.info('cand: {} is not legal.'.format(cand))
        #     return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True
    
    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))
    
    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])

            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler_num):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res
    
    def mutate_init_x(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in range(self.sampler_num):
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)

            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del(candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        original_num_steps = self.sampler_num
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))] 
        return use_timestep
    
    def get_cand_fid(self, cand=None, opt=None, device='cuda'):
        # t1 = time.time()
        
        
        # sample_time = time.time() - t1
        # # active model
        # t1 = time.time()
        # all_samples = np.array(all_samples)
        # all_samples = torch.Tensor(all_samples)
        # fid = calculate_fid(data1=all_samples,ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=320, dims=2048, device='cuda')

        # file_dir = "_".join([str(x) for x in cand])
        # cali_dir = os.path.join(self.opt.outdir, file_dir)
        torch.cuda.empty_cache()

        ckpt = './models/ldm/lsun_beds256/model.ckpt'
        base_configs = sorted(glob.glob('./models/ldm/lsun_beds256/config.yaml'))
        opt.base = base_configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        unknown = []
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        gpu = True
        eval_mode = True

        model, global_step = load_model(config, ckpt, gpu, eval_mode)
        model.ema_load()

        cand = cand[::-1]
        cali_data, cali_ts_list = get_calibration(cand)

        # import pdb
        # pdb.set_trace()

        # path = './out/lsun_beds256/samples/01900000/2023-09-19-mission-3/numpy/5000x256x256x3-samples.npz'
        # # path = './reference/VIRTUAL_lsun_bedroom256.npz'
        # all_samples = np.load(path)
        # all_samples = torch.tensor(all_samples['arr_0']).to(torch.float32).permute(0, 3, 1, 2)
        # all_samples = all_samples/255

        # all_samples = all_samples[0:1000, :, :, :]

        all_samples = self.get_image(model=model, cali_data=cali_data, cali_ts_list=cali_ts_list)

        all_samples = np.array(all_samples)
        all_samples = torch.tensor(all_samples)

        # import pdb
        # pdb.set_trace()

        fid = calculate_fid(data1=all_samples,ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=50, dims=2048, device='cuda')

        # cand = np.array(cand)
        # fid = np.sum(cand)

        logging.info('FID: ' + str(fid))

        del model

        # fid_time = time.time() - t1
        # logging.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid
    
    def search(self):
        logging.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)

        else:
            init_x = make_ddim_timesteps(ddim_discr_method=self.ddim_discretize, num_ddim_timesteps=self.time_step, num_ddpm_timesteps=self.sampler_num,verbose=False)
            init_x = list(init_x)

            self.is_legal_before_search(str(init_x))
            self.candidates.append(str(init_x))
            self.get_random_before_search(self.population_num // 2)

            res = self.mutate_init_x(x0=str(init_x), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
            self.candidates += res
        
        # self.rf_features = [eval(self.candidates[j]) for j in range(len(self.candidates))]
        # rf_features = np.asarray(self.rf_features, dtype='float')

        # try:    
        #     self.rf_lebal = [dec(self.vis_dict[self.candidates[j]]['fid']) for j in range(len(self.candidates))]
        #     self.RandomForestClassifier.fit(rf_features, self.rf_lebal)
        # except: 
        #     import pdb
        #     pdb.set_trace()

        while self.epoch < self.max_epochs:
            logging.info('epoch = {}'.format(self.epoch))
            
            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            
            self.update_top_k(self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logging.info('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):
                logging.info('No.{} {} fid = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['fid']))
            
            if self.epoch + 1 == self.max_epochs:
                break
            # sys.exit()
            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob)
            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num) #变异+杂交凑不足population size的部分重新随机采样

            # rf_features = np.asarray(self.rf_features, dtype='float')
            # self.RandomForestClassifier.fit(rf_features, self.rf_lebal) # refit

            self.epoch += 1
   
    def get_image(self, model, cali_data, cali_ts_list):
        opt = self.opt
        DM = model.model.diffusion_model

        if opt.split:
            setattr(DM, "split", True)

        # build quantization parameters
        wq_params = {'n_bits': opt.n_bits_w, 'channel_wise': opt.channel_wise, 'scale_method': 'max'}
        aq_params = {'n_bits': opt.n_bits_a, 'symmetric': opt.sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': opt.act_quant}
        QDM = QuantModel(model=DM, weight_quant_params=wq_params, act_quant_params=aq_params)
        QDM.cuda()
        QDM.eval()
        if not opt.disable_8bit_head_stem:
            print('Setting the first and the last layer to 8-bit')
            QDM.set_first_last_layer_to_8bit()

        device = next(QDM.parameters()).device

        
        # cali_data = cali_data[:500]
        # cali_ts_list = cali_ts_list[:500]

        # Initialize weight and activation quantization parameters
        QDM.set_quant_state(False, True)
        
        for name, module in DM.named_modules():
            if isinstance(module, BaseQuantBlock):
                module.get_name(name)
            elif isinstance(module, QuantModule):
                module.get_name(name)    
        
        print("Calibration begin!")
        QDM.set_init_ema()
        QDM.set_silu_asym()
        with torch.no_grad():
            # _ = QDM(cali_data[:opt.batch_size].to(device), cali_ts_list[:opt.batch_size].to(device))
            for i in range(int(cali_data.size(0) / opt.cali_batch_size)):
                print("cali_batch: %d" % i)
                _ = QDM(cali_data[i * opt.cali_batch_size:(i + 1) * opt.cali_batch_size].to(device), cali_ts_list[i * opt.cali_batch_size:(i + 1) * opt.cali_batch_size].to(device))
        QDM.set_calibrated()
        print("Calibration complete!")

        kwargs = dict(cali_data=cali_data, cali_ts_list=cali_ts_list, iters=opt.iters_a, act_quant=True, opt_mode='mse', lr=opt.lr, p=opt.p, batch_size=10)

        def recon_model(model: torch.nn.Module):
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    if module.ignore_reconstruction is True:
                        logging.info('Ignore reconstruction of layer {}'.format(module.name))
                        continue
                    else:
                        logging.info('Reconstruction for layer {}'.format(module.name))
                        layer_reconstruction(QDM, module, **kwargs)
                elif isinstance(module, BaseQuantBlock):
                    if module.ignore_reconstruction is True:
                        logging.info('Ignore reconstruction of block {}'.format(module.name))
                        continue
                    else:
                        logging.info('Reconstruction for block {}'.format(module.name))
                        block_reconstruction(QDM, module, **kwargs)
                else:
                    recon_model(module)
        
        QDM.set_zp_fixed()
        recon_model(QDM)

        QDM.set_quant_state(weight_quant=False, act_quant=True)
        # QDM.disable_network_output_quantization()

        with torch.no_grad():
            all_img = run(model, eta=opt.eta, vanilla=opt.vanilla_sample, n_samples=opt.n_samples,
                        custom_steps=opt.custom_steps, batch_size=opt.batch_size)

        logging.info("done.")

        return all_img

def main():
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    
    seed_all(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # log
    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    batch_size = opt.batch_size
    dpm_params = None

    ## build EA
    t = time.time()
    searcher = EvolutionSearcher(opt=opt, time_step=opt.time_step, ref_mu=opt.ref_mu, ref_sigma=opt.ref_sigma, sampler_num=200, batch_size=batch_size, dpm_params=dpm_params)
    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    main()
