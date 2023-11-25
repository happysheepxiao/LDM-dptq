import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

current_path = os.path.dirname(__file__)  # 获取当前需调用模块文件所在的目录
current_folder_path = os.path.dirname(current_path)  # 获取当前文件所在的文件夹所在的目录
sys.path.append(current_folder_path)  # 添加到环境变量中

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config



import pdb
import random
from quant_dptq import *

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

rescale = lambda x: (x + 1.) / 2.


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
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
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

    x_sample = model.decode_first_stage(sample)

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


def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    # n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    n_saved = len(glob.glob(os.path.join(logdir, '*.png')))

    # path = logdir
    if model.cond_stage_model is None:
        all_images = []


        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size + 1, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)            

            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            # ts = logs["ts"]
            if np_path is None:
                for x in batch:

                    # import pdb
                    # pdb.set_trace()

                    # img = custom_to_pil(x)
                    # # dir_name = os.path.join(path, f"{ts[n_saved].detach().cpu()}")
                    # # if not os.path.isdir(dir_name):
                    # #     os.makedirs(dir_name)
                    # # imgpath = os.path.join(path, f"{ts[n_saved].detach().cpu()}", f"{key}_{n_saved:06}.png")
                    # imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    # img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--seed', default=1000, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--cali_batch_size', default=25, type=int, help='mini-batch size for data loader')
    parser.add_argument('--recon_batch_size', default=5, type=int, help='mini-batch size for data loader')
    parser.add_argument('--train_batch_size', default=6, type=int, help='mini-batch size for data loader')
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
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')
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
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

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


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))

    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)

    model.ema_load()

    seed_all(opt.seed)

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

    # cand = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]
    cand = [1, 4, 7, 30, 46, 61, 64, 66, 77, 93, 127, 128, 130, 150, 151, 152, 156, 179, 182, 197]

    cand = cand[::-1]
    cali_data, cali_ts_list = get_calibration(cand)

    # Initialize weight and activation quantization parameters
    QDM.set_quant_state(True, True)
    
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

    print(QDM._modules.items())
    # import pdb
    # pdb.set_trace()

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, cali_ts_list=cali_ts_list, iters=opt.iters_w, weight=opt.weight, asym=True, b_range=(opt.b_start, opt.b_end), 
                  warmup=opt.warmup, act_quant=opt.act_quant, opt_mode='mse', lr=opt.lr, p=opt.p, recon_batch_size=opt.recon_batch_size, train_batch_size=opt.train_batch_size,
                  ratio=opt.ratio)
    # kwargs = dict(cali_data=cali_data, cali_ts_list=cali_ts_list, iters=opt.iters_a, act_quant=True, opt_mode='mse', lr=opt.lr, p=opt.p, batch_size=8)

    def recon_model(model: torch.nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(module.name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(module.name))
                    layer_reconstruction(QDM, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(module.name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(module.name))
                    block_reconstruction(QDM, module, **kwargs)
            else:
                recon_model(module)

    QDM.set_zp_fixed()
    recon_model(QDM)

    QDM.set_quant_state(weight_quant=True, act_quant=True)
    # QDM.disable_network_output_quantization()

    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")
    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)

    # print("Saving calibrated quantized UNet model")
    # for m in QDM.model.modules():
    #     if isinstance(m, AdaRoundQuantizer):
    #         m.zero_point = torch.nn.Parameter(m.zero_point)
    #         m.delta = torch.nn.Parameter(m.delta)
    #     elif isinstance(m, UniformAffineQuantizer):
    #         if m.zero_point is not None:
    #             if not torch.is_tensor(m.zero_point):
    #                 m.zero_point = torch.nn.Parameter(torch.tensor(float(m.zero_point)))
    #             else:
    #                 m.zero_point = torch.nn.Parameter(m.zero_point)
    # torch.save(QDM.state_dict(), os.path.join(logdir, "ckpt.pth"))         


    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    with torch.no_grad():
        run(model, imglogdir, eta=opt.eta,
            vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
            batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")
