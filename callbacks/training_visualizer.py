import os
import pytorch_lightning as pl
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb

def unnorm(x):
    '''convert from range [-1, 1] to [0, 1]'''
    return (x+1) / 2

def clip_image(x, min=0., max=1.):
    return torch.clamp(x, min=min, max=max)

def format_dtype_and_shape(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        if len(x.shape) == 4 and x.shape[1] == 3:
            x = x.permute(0, 2, 3, 1)
        x = x.detach().cpu().numpy()
    return x

def tensor2image(x):
    x = x.float() # handle bf16
    '''convert 4D (b, dim, h, w) pytorch tensor to wandb Image class'''
    grid_img = torchvision.utils.make_grid(
        x, nrow=4
    ).permute(1, 2, 0).detach().cpu().numpy()
    img = wandb.Image(
        grid_img
    )
    return img

class VAETrainingLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            image_recon = tensor2image(clip_image(
                outputs['image_recon'][:self.max_num_images]
            ))
            self.wandb_logger.experiment.log({
                'train/input_image': input_image,
                'train/image_recon': image_recon,
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            image_recon = tensor2image(clip_image(
                outputs['image_recon'][:self.max_num_images]
            ))
            self.wandb_logger.experiment.log({
                'val/input_image': input_image,
                'val/image_recon': image_recon,
            })

class DiffusionTrainingLogger(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger=None,
        max_num_images: int=16,
    ) -> None:
        super().__init__()
        self.wandb_logger = wandb_logger
        self.max_num_images = max_num_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 1000 == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            edge = tensor2image(clip_image(
                batch['edge'][:self.max_num_images]
            ))
            edge_pred = tensor2image(clip_image(
                outputs['edge_pred'][:self.max_num_images]
            ))
            if 'focal_region' in outputs:
                focal_region = outputs['focal_region'][:self.max_num_images]
                focal_region = torch.concat([focal_region, focal_region, focal_region], dim=1)
                focal_region = tensor2image(clip_image(
                    focal_region
                ))
                self.wandb_logger.experiment.log({
                    'train/focal_region': focal_region,
                })
            self.wandb_logger.experiment.log({
                'train/input_image': input_image,
                'train/edge': edge,
                'train/edge_pred': edge_pred,
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            edge = tensor2image(clip_image(
                batch['edge'][:self.max_num_images]
            ))
            edge_pred = outputs['edge_pred'][:self.max_num_images] # this is B, h, w, c
            edge_pred = tensor2image(clip_image(
                torch.from_numpy(edge_pred).permute(0, 3, 1, 2)
            ))
            edge_pred_no_cfg = outputs['edge_pred_no_cfg'][:self.max_num_images] # this is B, h, w, c
            edge_pred_no_cfg = tensor2image(clip_image(
                torch.from_numpy(edge_pred_no_cfg).permute(0, 3, 1, 2)
            ))
            self.wandb_logger.experiment.log({
                'val/input_image': input_image,
                'val/edge': edge,
                'val/edge_pred': edge_pred,
                'val/edge_pred_no_cfg': edge_pred_no_cfg,
            })
