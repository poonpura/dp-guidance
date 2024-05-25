import os
import errno
import argparse
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision import utils

import classifier.sotonami as sotonami
from helper import OptimizerDetails
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def get_optimation_details(args, device):
    guidance_func = (sotonami.ClassifierModule()
                    .load_from_checkpoint(args.classifier_ckpt)).to(device)
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = torch.nn.CrossEntropyLoss()

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff
    operation.classic = args.classic_guidance

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation


def main(opt):
    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    sampler = DDIMSamplerWithGrad(model)
    operation = get_optimation_details(opt, device)
    batch_size = opt.batch_size
    target_label = torch.tensor([opt.target_label]).to(device)
    torch.set_grad_enabled(False)
    prompt = opt.text

    uc = None
    if opt.scale != 1.0:
        uc = model.module.get_learned_conditioning(batch_size * [""])
    c = model.module.get_learned_conditioning(batch_size * [prompt])

    for multiple_tries in range(opt.trials):
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
        samples_ddim, start_zt = sampler.sample(S=opt.ddim_steps,
                                         conditioning=c,
                                         batch_size=batch_size,
                                         shape=shape,
                                         verbose=False,
                                         unconditional_guidance_scale=opt.scale,
                                         unconditional_conditioning=uc,
                                         eta=opt.ddim_eta,
                                         operated_image=target_label,
                                         operation=operation)
        x_samples_ddim = model.module.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        utils.save_image(x_samples_ddim, f'{results_folder}/new3_img_{multiple_tries}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt", default=5.0, type=float)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[1], type=int)
    parser.add_argument("--text", default=None)
    parser.add_argument("--trials", default=5, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--classifier_ckpt", default="", type=str)
    parser.add_argument("--target_label", default=0, type=int)
    parser.add_argument("--classic_guidance", action="store_true")

    args = parser.parse_args()
    main(args)