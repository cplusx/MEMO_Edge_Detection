import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
from misc_utils.model_utils import instantiate_from_config, get_obj_from_str
from einops import rearrange, repeat
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from typing import Union
from models.UNet_with_dinov2 import UNet2DwithDINOv2
from peft import LoraConfig
from peft.mapping import inject_adapter_in_model

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L236
    Only for the Flow Matching models
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u

def truncated_linear_timestep_sampling(
    batch_size: int, 
    truncate_at: float, 
    min_sample_rate: float, 
    device=None, 
    dtype=torch.float32
) -> torch.Tensor:
    """
    Draw `batch_size` i.i.d. samples x ∈ [0, 1] from the 1-D PDF

        r(x) = { max_r - (max_r-min_r)·x/t          , 0 ≤ x < t
               { min_r                              , t ≤ x ≤ 1

    where `t = truncate_at`, `min_r = min_sample_rate`, and `max_r`
    is chosen so the PDF integrates to 1, i.e.
        1 = min_r + ½·(max_r - min_r)·t
        → max_r = min_r + 2(1 - min_r)/t .

    After `x ≥ t` the density is flat (constant `min_r`).

    Parameters
    ----------
    batch_size      number of samples to return
    truncate_at     t ∈ (0, 1]; end of the linear section
    min_sample_rate min_r ∈ (0, 1); floor of the density
    device, dtype   forwarded to `torch.rand`

    Returns
    -------
    Tensor of shape (batch_size,) with values in [0, 1].
    """
    if not (0 < truncate_at <= 1):
        raise ValueError("truncate_at must be in (0, 1]")
    if not (0 < min_sample_rate < 1):
        raise ValueError("min_sample_rate must be in (0, 1)")

    t, min_r = truncate_at, min_sample_rate
    max_r = min_r + 2 * (1 - min_r) / t          # from normalisation
    slope = (max_r - min_r) / t                  # positive

    # Pre-compute CDF at the truncation point (area of the left section)
    A = max_r * t - 0.5 * slope * t ** 2         # ∫0^t r(x) dx

    u = torch.rand(batch_size, device=device, dtype=dtype)
    x = torch.empty_like(u)

    left_mask = u < A                     # samples that land in the linear part
    u_left = u[left_mask]

    # Invert quadratic CDF:  u = max_r·x – ½·slope·x²
    # Pick the root near 0 (minus sign).
    x[left_mask] = (max_r - torch.sqrt(max_r ** 2 - 2 * slope * u_left)) / slope

    # Right (flat) part:  CDF is affine with slope = min_r
    u_right = u[~left_mask]
    x[~left_mask] = t + (u_right - A) / min_r

    return x


def focal_mask_sampling(
    edge_map: torch.Tensor,  # (B,H,W) – edges = 1, bg = 0
    t:        torch.Tensor,   # (B,)    – sampling ratio 0<t<1
    edge_weights: Union[float,int,str] = 'auto',
) -> torch.Tensor:       # (B,H,W) – sampling mask
    """
    Per image we invert the edge:bg ratio (1:k  →  k:1) and keep exactly
    t[i]*H*W pixels in total.

    No Python loop over the batch; works on CPU or GPU.
    """
    B, H, W = edge_map.shape
    N   = H * W
    eps = 1e-8                         # numerical safety

    edge_flat = edge_map.reshape(B, N)
    num_edge  = edge_flat.sum(1).float()          # (B,)
    num_bg    = N - num_edge                     # (B,)

    # --- weights that invert the original ratio ---------------------------
    if edge_weights == 'auto':
        w_edge = num_bg                  # large if bg dominates
        w_bg   = num_edge                # large if edges dominate
        weights = edge_flat * w_edge[:, None] + (1 - edge_flat) * w_bg[:, None]
    elif isinstance(edge_weights, (int, float)):
        w_edge = edge_weights
        w_bg   = 1
        weights = edge_flat * w_edge + (1 - edge_flat) * w_bg
    weights = weights + eps          # make sure no row is all-zero

    # How many pixels do we want from each image?
    k_per_img = (t * N).long()       # (B,)

    # ----------------------------------------------------------------------
    #  Group images that share the same k so we can call torch.multinomial
    #  once per group instead of once per sample.  In practice there are
    #  only a handful of different k’s, so this is still close to fully
    #  vectorised and 100 % artefact-free.
    # ----------------------------------------------------------------------
    sampled = torch.zeros_like(edge_flat, dtype=torch.float32)

    for k in k_per_img.unique():
        if k == 0:
            continue
        rows = (k_per_img == k).nonzero(as_tuple=False).squeeze(1)
        w    = weights[rows]                    # (|rows|, N)
        idx  = torch.multinomial(w, k.item(), replacement=False)  # (|rows|, k)
        batch_rows = rows.unsqueeze(1).expand_as(idx)             # same shape
        sampled[batch_rows, idx] = 1

    return sampled.view(B, H, W) > 0

def get_focal_weight(target, dilate=2, scale='auto'):
    # target: B, H, W
    # Determine the edge region
    edge_region = (target > 1e-4).unsqueeze(1)  # Convert to B, 1, H, W

    # Apply dilation to the edge region
    kernel_size = 2 * dilate + 1
    dilation_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=target.device, dtype=torch.float32)
    edge_region = edge_region.float()
    focal_region = F.conv2d(edge_region, dilation_kernel, padding=dilate) > 0  # Dilate

    # Convert to float and interpolate to match prediction size
    focal_region = focal_region.float()
    focal_region = F.interpolate(focal_region, size=target.shape[-2:], mode='bilinear')
    focal_region = (focal_region > 1e-4).float()

    # Compute weights
    if scale == 'auto':
        h, w = focal_region.shape[-2:]
        scale = h * w / (focal_region.sum(dim=[-2, -1], keepdim=True) + 1e-4)
        # print('scale', scale)
        weights = focal_region * (scale - 1) + 1.0
    else:
        weights = focal_region * (scale - 1) + 1.0
    weights /= weights.mean(dim=[-2, -1], keepdim=True)  # Normalize weights

    return weights.squeeze(1), focal_region.squeeze(1)

def get_multiclass_focal_weight(target):
    # target: B, H, W
    # NOTE: when using this function, one should use sum() along the batch size when computing the loss
    num_classes = target.max().item() + 1  # Assuming classes are 0-indexed
    num_pixel_per_class = target.view(-1).to(torch.long).bincount(minlength=num_classes)
    weights = torch.zeros_like(target, dtype=torch.float32)
    for cls_id, cls_num in enumerate(num_pixel_per_class):
        weights = torch.where(
            target == cls_id,
            1 / (num_classes * cls_num + 1e-4),
            weights
        )
    return weights

def edge_consistency_loss(pred: torch.Tensor, edge_labels: torch.Tensor, reduction='mean', valid_mask=None, max_labels=30):
    '''
    pred: (B, C, H, W)
    edge_labels: (B, H, W)
    valid_mask: (B, H, W)
    '''
    unique_labels = torch.unique(edge_labels)[1:1+max_labels] # exclude 0, the background, limit to max_labels
    connect_component_mask = (edge_labels[..., None] == unique_labels).float() # (B, H, W, num_labels)
    connect_component_mask = rearrange(connect_component_mask, 'b h w n -> b 1 h w n')

    prob = torch.softmax(pred, dim=1) # (B, C, H, W)
    prob = rearrange(prob, 'b c h w -> b c h w 1')
    log_prob = torch.log(prob + 1e-6)

    masked_prob = prob * connect_component_mask # (B, C, H, W, num_labels)
    masked_log_prob = log_prob * connect_component_mask # (B, C, H, W, num_labels)
    avg_log_prob = masked_log_prob.sum(dim=(2, 3), keepdim=True) / (connect_component_mask.sum(dim=(2, 3), keepdim=True) + 1e-6) # (B, C, num_labels)

    jsd = torch.sum(masked_prob * (masked_log_prob - avg_log_prob.detach()), dim=(1, 4)) # (B, H, W)

    if reduction == 'mean':
        jsd = jsd.sum(dim=(1, 2)) / (connect_component_mask.sum(dim=(1, 2, 3, 4)) + 1e-6) # (B,)
        return jsd.mean()
    elif reduction == 'sum':
        jsd = jsd.sum(dim=(1, 2)) / (connect_component_mask.sum(dim=(1, 2, 3, 4)) + 1e-6) # (B,)
        return jsd.sum()
    elif reduction == 'masked':
        jsd = (valid_mask * jsd).sum(dim=(1, 2)) / (connect_component_mask.sum(dim=(1, 2, 3, 4)) + 1e-6) # (B,)
        return jsd.mean()

class MEMOEdgeTrainer(pl.LightningModule):
    def __init__(
        self,
        pipe, 
        guidance_scale: float=4.0,
        num_inference: int=20,
        optim_args: dict={},
        loss_weights: dict={
            'focal_xentropy': 1.0,
        },
        gradient_checkpointing: bool=False,
        use_8bit_adam: bool=False,
        accumulate_grad_batches: int=8,
        use_ema: bool=False,
        ema_decay: float=0.99,
        ema_start: int=100,
        cond_drop_rate: float=0.1,
        timestep_sampling_weighting_scheme: str = 'uniform',
        timestep_sampling_kwargs: dict = {},
        mask_strategy: str = 'uniform',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pipe = pipe
        self.denoiser: UNet2DwithDINOv2 = pipe.denoiser
        self.guidance_scale = guidance_scale
        self.num_inference = num_inference
        self.optim_args = optim_args
        self.loss_weights = loss_weights
        self.use_8bit_adam = use_8bit_adam
        self.cond_drop_rate = cond_drop_rate
        self.accumulate_grad_batches = accumulate_grad_batches
        self.timestep_sampling_weighting_scheme = timestep_sampling_weighting_scheme
        self.timestep_sampling_kwargs = timestep_sampling_kwargs
        self.mask_strategy = mask_strategy
        if self.accumulate_grad_batches > 1:
            self.automatic_optimization = False
        if use_ema:
            self.use_ema = use_ema
            self.ema_decay = ema_decay
            self.ema_start = ema_start
            self.init_ema_model()

        self.call_save_hyperparameters()


        if gradient_checkpointing:
            if self.denoiser._supports_gradient_checkpointing:
                self.denoiser.enable_gradient_checkpointing()

    def init_ema_model(self):
        if self.ema_decay:
            self.ema_denoiser = AveragedModel(self.denoiser, multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay))
            if self.local_rank == 0:
                print('INFO: EMA model enabled with decay', self.ema_decay)
            self.denoiser_state_dict = None


    def call_save_hyperparameters(self):
        self.save_hyperparameters(
            ignore=['ema_denoiser', 'denoiser', 'pipe']
        )

    def mask_target(self, edge, mask_ratio):
        # edge shape: (B, H, W)
        # mask: True for masked pixels, False for unmasked pixels
        if self.mask_strategy == 'uniform':
            rng = torch.rand(edge.shape, device=edge.device)
            mask = rng < mask_ratio.unsqueeze(-1).unsqueeze(-1)  # (B, H, W)
            edge[mask] = self.denoiser.mask_token_id
        elif self.mask_strategy == 'focal':
            _, focal_region = get_focal_weight(target=edge, dilate=1, scale='auto')  # (B, H, W)
            mask = focal_mask_sampling(
                edge_map=focal_region,
                t=mask_ratio,
                edge_weights=4 # empirically, at 4, the edges are mostly masked at t=0.6
            ).to(edge.device)  # (B, H, W)
            edge[mask] = self.denoiser.mask_token_id
        else:
            raise ValueError(f'Unknown mask strategy: {self.mask_strategy}')
        return edge, mask

    def get_mask_ratio(self, batch_size):
        if self.timestep_sampling_weighting_scheme == 'uniform':
            return (torch.rand(batch_size, device=self.device) * 1.01).clamp(0.001, 1) # * 1.01 so that there is 1% of chance that the mask ratio is 1.0
        elif self.timestep_sampling_weighting_scheme == 'logit_normal':
            if self.global_step == 0 and self.local_rank == 0:
                print(f'INFO: Using logit normal sampling for timestep sampling with mean=0.0 and std=1.0')
            logit_mean = 0.0
            logit_std = 1.0
            timesteps = compute_density_for_timestep_sampling(
                self.timestep_sampling_weighting_scheme,
                batch_size,
                logit_mean=logit_mean,
                logit_std=logit_std
            ) * 1.01
            return timesteps.clamp(0.001, 1)  # Ensure mask ratio is between 0.03 and 1
        elif self.timestep_sampling_weighting_scheme == 'truncated_linear':
            truncate_at = self.timestep_sampling_kwargs.get('truncate_at', 0.6)
            min_sample_rate = self.timestep_sampling_kwargs.get('min_sample_rate', 0.7)
            timesteps = truncated_linear_timestep_sampling(
                batch_size, 
                truncate_at=truncate_at, 
                min_sample_rate=min_sample_rate,
                device=self.device,
                dtype=torch.float32
            ) * 1.01
            return timesteps.clamp(0.001, 1)
        else:
            raise ValueError(f'Unknown timestep sampling weighting scheme: {self.timestep_sampling_weighting_scheme}')

    def train_internal_step(self, batch, batch_idx, mode='train'):
        image = batch['image'] # should be in [0, 1], (B, C, H, W)
        edge = batch['edge'] # should be in (B, H, W), long or int
        image_cond = image.to(self.denoiser.dtype)
        edge = edge.to(torch.long)

        mask_ratio = self.get_mask_ratio(len(image)).to(edge.device)
        masked_edge, mask = self.mask_target(edge.clone(), mask_ratio)
        # mask: True for masked pixels, False for unmasked pixels

        # randomly drop some condition
        rng = torch.rand(image_cond.shape[0], device=image_cond.device)
        drop_mask = rng < self.cond_drop_rate
        image_cond[drop_mask] = 0.0

        edge_pred = self.denoiser(
            masked_edge, 
            mask_ratio, 
            image_cond=image_cond,
            return_dict=False
        )[0]

        loss = 0
        res_dict = {
            'image': image,
            'edge': edge,
            'edge_pred': edge_pred,
        }
        if 'xentropy' in self.loss_weights:
            x_loss = F.cross_entropy(
                edge_pred,
                edge,
                reduction='none'
            ) # B, H, W
            x_loss = (x_loss * mask.float()).sum(dim=[1, 2]) / mask.float().sum(dim=[1, 2])  # average over pixels
            # x_loss = (x_loss / mask_ratio).mean()  # average over batch
            x_loss = (x_loss * (1 / mask_ratio).clamp(1, 5)).mean()  # average over batch
            x_loss = x_loss * self.loss_weights['xentropy']
            self.log(f'{mode}/xentropy', x_loss, sync_dist=True)
            res_dict['xentropy'] = x_loss.item()
            loss += x_loss

        if 'focal_xentropy' in self.loss_weights:
            focal_weights, focal_region = get_focal_weight(target=edge, dilate=1, scale='auto')  # (B, H, W)
            f_x_loss = F.cross_entropy(
                edge_pred,
                edge,
                reduction='none',
            )
            f_x_loss = (f_x_loss * focal_weights * mask.float()).sum(dim=[1, 2]) / (focal_weights * mask.float()).sum(dim=[1, 2])
            # f_x_loss = (f_x_loss / mask_ratio).mean()  # average over batch
            f_x_loss = (f_x_loss * (1 / mask_ratio).clamp(1, 5)).mean()  # average over batch
            f_x_loss = f_x_loss * self.loss_weights['focal_xentropy']
            self.log(f'{mode}/focal_xentropy', f_x_loss, sync_dist=True)
            res_dict['focal_xentropy'] = f_x_loss.item()
            loss += f_x_loss

        if 'multiclass_focal_xentropy' in self.loss_weights:
            focal_weights = get_multiclass_focal_weight(edge)
            m_f_x_loss = F.cross_entropy(
                edge_pred,
                edge,
                reduction='none',
            )
            m_f_x_loss = (m_f_x_loss * focal_weights * mask.float()).sum(dim=[1, 2]) / (focal_weights * mask.float()).sum(dim=[1, 2])
            m_f_x_loss = (m_f_x_loss * (1 / mask_ratio).clamp(1, 5)).sum()  # when using get_multiclass_focal_weight, sum over batch since the focal weights are comptued globally
            m_f_x_loss = m_f_x_loss * self.loss_weights['multiclass_focal_xentropy']
            self.log(f'{mode}/multiclass_focal_xentropy', m_f_x_loss, sync_dist=True)
            res_dict['multiclass_focal_xentropy'] = m_f_x_loss.item()
            loss += m_f_x_loss

        if 'consistency' in self.loss_weights:
            if 'edge_index' in batch:
                # laion edge v2 naming
                edge_index = batch['edge_index']  # (B, H, W)
            else:
                # laion edge v1 naming
                edge_index = batch['edge_labels']
            consistency_loss = edge_consistency_loss(edge_pred, edge_index, reduction='mean')
            consistency_loss = consistency_loss * self.loss_weights['consistency']
            res_dict['consistency_loss'] = consistency_loss.item()

            loss += consistency_loss
            self.log(f'{mode}_consistency_loss', consistency_loss, sync_dist=True)
            res_dict['consistency_loss'] = consistency_loss.item()

        with torch.no_grad():
            acc = (edge_pred.argmax(dim=1) == edge).float().mean()
        self.log(f'{mode}/acc', acc, sync_dist=True)

        self.log(f'{mode}/loss', loss, sync_dist=True)
        res_dict['loss'] = loss

        return res_dict

    def training_step(self, batch, batch_idx):
        N = self.accumulate_grad_batches
        if self.global_step % 100 == 0 and self.local_rank == 0:
            print(f'INFO: global step {self.global_step}')
        if N == 1:
            res_dict = self.train_internal_step(batch, batch_idx, mode='train')
            if self.use_ema and self.global_step > self.ema_start:
                if (self.local_rank == 0) and (self.global_step % 100 == 0):
                    print(f'INFO: updating EMA model @ step {self.global_step}')
                self.ema_denoiser.update_parameters(self.denoiser)
            return res_dict

        # accumulate gradients with manual optimization (for compatibility with gradient checkpointing)
        opt = self.optimizers()
        res_dict = self.train_internal_step(batch, batch_idx, mode='train')
        loss = res_dict['loss'] / N

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

        if (batch_idx + 1) % N == 0:
            opt.step()
            opt.zero_grad()

            if self.use_ema and self.global_step > self.ema_start:
                if (self.local_rank == 0) and (self.global_step % (N * 100) == 0):
                    print(f'INFO: updating EMA model @ step {self.global_step}')
                self.ema_denoiser.update_parameters(self.denoiser)
        
        return res_dict

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        image = batch['image'] # should be in [0, 1], (B, C, H, W)
        edge = batch['edge']
        generated = self.pipe(
            images=image,
            max_inference_steps=self.num_inference,
            guidance_scale=1.0,
        )['edges']

        generated_with_guide = self.pipe(
            images=image,
            max_inference_steps=self.num_inference,
            guidance_scale=self.guidance_scale,
        )['edges']

        acc = (generated_with_guide == edge.cpu().numpy()).astype(np.float32).mean()
        acc_no_cfg = (generated == edge.cpu().numpy()).astype(np.float32).mean()

        return {
            'image': image,
            'edge': edge,
            'edge_pred_no_cfg': generated,
            'edge_pred': generated_with_guide,
            'acc': acc,
            'acc_no_cfg': acc_no_cfg,
        }

    def on_train_epoch_end(self):
        if self.use_ema and self.global_step > self.ema_start:
            self.denoiser_state_dict = self.denoiser.state_dict()
            ema_state_dict = self.ema_denoiser.module.state_dict()
            self.denoiser.load_state_dict(ema_state_dict)

    def on_train_epoch_start(self):
        if self.use_ema and self.denoiser_state_dict is not None and self.global_step > self.ema_start:
            self.denoiser.load_state_dict(self.denoiser_state_dict)

    def configure_optimizers(self):
        if self.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            self.denoiser.parameters(),
            **self.optim_args
        )
        return optimizer

class MEMOEdgeLoRATrainer(MEMOEdgeTrainer):
    def load_denoiser_weights(self, init_weights):
        model_weights = torch.load(init_weights, map_location='cpu')
        if 'module' in model_weights:
            model_weights = model_weights['module']
        ema_model_weights = {k.replace('ema_denoiser.module.', ''): v for k, v in model_weights.items() if 'ema_denoiser.module.' in k}
        self.denoiser.load_state_dict(ema_model_weights)
        print("Loaded model weights from:", init_weights)
        
    def __init__(self, *args, init_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert init_weights, "init_weights must be provided"

        # # load weights
        # model_weights = torch.load(init_weights, map_location='cpu')
        # if 'module' in model_weights:
        #     model_weights = model_weights['module']
        # ema_model_weights = {k.replace('ema_denoiser.module.', ''): v for k, v in model_weights.items() if 'ema_denoiser.module.' in k}
        # self.denoiser.load_state_dict(ema_model_weights)
        # print("Loaded model weights from:", init_weights)

        if not 'lora' in init_weights:
            print('Loading full model weights before injecting LoRA adapter.')
            self.load_denoiser_weights(init_weights)

        unet_lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "conv1", "conv2", "qkv", 'fc1', 'fc2'],
        )
        self.denoiser = inject_adapter_in_model(unet_lora_config, self.denoiser, adapter_name='unet_lora')
        print("Injected LoRA adapter into UNet model.")

        if 'lora' in init_weights:
            print('Loading weights after injecting LoRA adapter.')
            self.load_denoiser_weights(init_weights)

        # reinit the ema model
        if self.use_ema:
            self.init_ema_model()

    def configure_optimizers(self):
        optimizer_cls = torch.optim.AdamW

        trainable_params = []
        for name, param in self.denoiser.named_parameters():
            if 'lora' in name or 'conv_out' in name:
                trainable_params.append(param)
        print("Trainable parameters:", len(trainable_params))

        optimizer = optimizer_cls(
            trainable_params,
            **self.optim_args
        )
        return optimizer
