import os
import pytorch_lightning as pl
import torch
import torchvision
import cv2
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb
from .training_visualizer import tensor2image, clip_image
from edge_datasets.edge_datasets.LAION_synthetic import colors_panel_36

def map_edge_to_color(edge, colors_panel):
    # edge: (B, H, W)
    # colors_panel: (N, 3)
    edge = torch.tensor(edge).to(torch.int16)
    B, H, W = edge.shape
    edge = edge.view(-1)
    colors = torch.from_numpy(colors_panel[edge])
    colors = colors.view(B, H, W, 3).permute(0, 3, 1, 2) / 255.0
    return colors

class MEMOTrainingLogger(Callback):
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
            edge = batch['edge'].cpu().detach()
            color_edge = tensor2image(clip_image(
                map_edge_to_color(edge[:self.max_num_images], colors_panel_36)
            ))
            self.wandb_logger.experiment.log({
                'train/input_image': input_image,
                'train/edge': color_edge,
            })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        acc = outputs['acc']
        self.val_acc_list.append(acc)
        acc_no_cfg = outputs['acc_no_cfg']
        self.val_acc_no_cfg_list.append(acc_no_cfg)
        if batch_idx == 0:
            input_image = tensor2image(clip_image(
                batch['image'][:self.max_num_images]
            ))
            edge = batch['edge'].cpu().detach()
            color_edge = tensor2image(clip_image(
                map_edge_to_color(edge[:self.max_num_images], colors_panel_36)
            ))

            edge_pred = outputs['edge_pred']
            color_edge_pred = tensor2image(clip_image(
                map_edge_to_color(edge_pred[:self.max_num_images], colors_panel_36)
            ))

            edge_pred_no_cfg = outputs['edge_pred_no_cfg']
            color_edge_pred_no_cfg = tensor2image(clip_image(
                map_edge_to_color(edge_pred_no_cfg[:self.max_num_images], colors_panel_36)
            ))
            self.wandb_logger.experiment.log({
                'val/input_image': input_image,
                'val/edge': color_edge,
                'val/edge_pred': color_edge_pred,
                'val/edge_pred_no_cfg': color_edge_pred_no_cfg,
            })

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_acc_list = []
        self.val_acc_no_cfg_list = []
        return super().on_validation_epoch_start(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_acc = np.mean(self.val_acc_list)
        val_acc_no_cfg = np.mean(self.val_acc_no_cfg_list)
        pl_module.log('val/acc', val_acc, sync_dist=True)
        pl_module.log('val/acc_no_cfg', val_acc_no_cfg, sync_dist=True)
        return super().on_validation_epoch_end(trainer, pl_module)