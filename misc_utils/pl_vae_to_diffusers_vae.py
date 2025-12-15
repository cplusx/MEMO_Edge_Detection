import argparse
import torch
from diffusers import AutoencoderKL

parser = argparse.ArgumentParser()
parser.add_argument('--pl_ckpt', type=str, required=True)
parser.add_argument('--save_name', type=str, required=True)

args = parser.parse_args()

vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='vae')
vae_ckpt = args.pl_ckpt
vae_ckpt = torch.load(vae_ckpt, map_location='cpu')['module']

ema_vae_ckpt = {k.replace('ema_vae.module.', ''): v for k, v in vae_ckpt.items() if 'ema_vae.module.' in k}
vae.load_state_dict(ema_vae_ckpt)
vae.save_pretrained(args.save_name)