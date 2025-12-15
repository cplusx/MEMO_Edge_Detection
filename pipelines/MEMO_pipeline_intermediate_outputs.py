from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from models.UNet_with_dinov2 import UNet2DwithDINOv2
from einops import rearrange
from pipelines.MEMO_pipeline import add_gumbel_noise

class MEMOEdgeWithIntermediatesPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "denoiser"

    def __init__(
        self,
        denoiser: UNet2DwithDINOv2,
    ):
        super().__init__()
        self.register_modules(denoiser=denoiser)

    def get_num_transfer_tokens(self, mask_index, steps, denoise_mode='uniform', denoise_kwargs=None):
        '''
        copied from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py#L22
        In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
        Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
        the expected number of tokens transitioned at each step should be consistent.

        This function is designed to precompute the number of tokens that need to be transitioned at each step.
        '''
        mask_num = mask_index.sum(dim=[1, 2])
        if denoise_mode == 'uniform':
            base = mask_num // steps
            remainder = mask_num % steps

            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64) + base.unsqueeze(1)

            for i in range(mask_num.size(0)):
                num_transfer_tokens[i, :remainder[i]] += 1
        elif denoise_mode == 'linear':
            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64)
            for b in range(mask_num.size(0)):
                N = mask_num[b].item()
                xs = torch.linspace(0, N, steps + 1, device=mask_index.device)
                mask_ratio_at_x = 1 - (-(xs / N)**2 + 2*xs/N)
                mask_remain_at_x = (N * mask_ratio_at_x).to(torch.int64)
                mask_remain_at_x[0] = N
                mask_remain_at_x[-1] = 0
                num_transfer_tokens[b] = mask_remain_at_x[0:-1] - mask_remain_at_x[1:]
        elif denoise_mode == 'exponential':
            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64)
            for b in range(mask_num.size(0)):
                N = mask_num[b].item()
                this_N = N / 2
                for i in range(steps):
                    num_transfer_tokens[b, i] = int(this_N)
                    this_N = this_N / 2
                    if this_N < 1:
                        break
                num_transfer_tokens[b, -1] += N - num_transfer_tokens[b].sum()
        elif denoise_mode == 'brachistochrone' or denoise_mode == 'cycloid':
            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64)
            for b in range(mask_num.size(0)):
                N = mask_num[b].item()
                ts = torch.linspace(0, np.pi, steps+1, device=mask_index.device)
                mask_ratio_at_t = (torch.cos(ts) + 1) / 2
                # print(mask_ratio_at_t)
                num_transfer_tokens[b] = (N * (mask_ratio_at_t[0:-1] - mask_ratio_at_t[1:])).to(torch.int64)
                num_transfer_tokens[b, -1] += N - num_transfer_tokens[b].sum()
        elif denoise_mode == 'circle_x':
            # y = 1 - sqrt(1 - (x-1)^2)
            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64)
            for b in range(mask_num.size(0)):
                N = mask_num[b].item()
                xs = torch.linspace(0, 1, steps + 1, device=mask_index.device)
                mask_ratio_at_x = 1 - torch.sqrt(1 - (xs - 1)**2)
                mask_remain_at_x = (N * mask_ratio_at_x).to(torch.int64)
                mask_remain_at_x[0] = N
                mask_remain_at_x[-1] = 0
                num_transfer_tokens[b] = mask_remain_at_x[0:-1] - mask_remain_at_x[1:]
                # print(mask_ratio_at_x)
        elif denoise_mode == 'circle_t':
            # y = 1 - sin(t) t \in [0, pi/2]
            num_transfer_tokens = torch.zeros((mask_num.size(0), steps), device=mask_index.device, dtype=torch.int64)
            for b in range(mask_num.size(0)):
                N = mask_num[b].item()
                ts = torch.linspace(0, np.pi / 2, steps + 1, device=mask_index.device)
                mask_ratio_at_t = 1 - torch.sin(ts)
                num_transfer_tokens[b] = (N * (mask_ratio_at_t[0:-1] - mask_ratio_at_t[1:])).to(torch.int64)
                num_transfer_tokens[b, -1] += N - num_transfer_tokens[b].sum()
        else:
            raise NotImplementedError(f"denoise_mode {denoise_mode} is not implemented.")

        return num_transfer_tokens

    @torch.no_grad()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        temperature: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        denoise_mode: str = 'uniform',
        denoise_kwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Union[Dict, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        image_cond = images.to(self.device)
        image_h, image_w = image_cond.shape[-2:]
        mask_token_id = self.denoiser.config.mask_token_id

        masked_edges = torch.ones(
            (batch_size, image_h, image_w),
            dtype=torch.long,
            device=self._execution_device,
        ) * mask_token_id

        mask_index = (masked_edges == mask_token_id)
        transfer_tokens = self.get_num_transfer_tokens(mask_index, num_inference_steps, denoise_mode=denoise_mode, denoise_kwargs=denoise_kwargs)
        num_masks = mask_index.sum(dim=[1, 2]).view(-1, 1)  # (B, 1)
        mask_ratio_list = torch.flip(transfer_tokens, dims=(1, )).float().cumsum(dim=1) / num_masks.float()  # (B, steps)
        mask_ratio_list = torch.flip(mask_ratio_list, dims=(1, )) # (steps, B), should be 1 -> 0
        # print(f"mask_ratio_list: {mask_ratio_list}")

        intermediate_masked_edges = []
        intermeidate_transfer_index = []
        intermediate_predict_confidence = []

        for i in self.progress_bar(range(num_inference_steps)):
            mask_index = (masked_edges == mask_token_id)
            mask_ratio = torch.tensor(mask_ratio_list[:, i], device=self.device)
            # mask_ratio = mask_ratio.expand(image_cond.shape[0])

            if guidance_scale > 1.0:
                masked_edges_input = torch.cat([masked_edges, masked_edges], dim=0)
                image_cond_input = torch.cat([
                    image_cond, torch.zeros_like(image_cond)
                ], dim=0)
                mask_ratio_input = torch.cat([mask_ratio, mask_ratio], dim=0)
            else:
                masked_edges_input = masked_edges
                image_cond_input = image_cond
                mask_ratio_input = mask_ratio

            edge_logits = self.denoiser(
                masked_edges_input,
                mask_ratio_input,
                image_cond=image_cond_input,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                edge_logits_cond, edge_logits_uncond = edge_logits.chunk(2, dim=0)
                edge_logits = edge_logits_uncond + guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = add_gumbel_noise(edge_logits, temperature=temperature)  # add Gumbel noise for sampling

            edge_logits = rearrange(edge_logits, 'b c h w -> b (h w) c')
            masked_edges = rearrange(masked_edges, 'b h w -> b (h w)')

            x_0 = torch.argmax(edge_logits, dim=-1)
            p = F.softmax(edge_logits, dim=-1)
            x_0_p = torch.gather(p, -1, x_0.unsqueeze(-1)).squeeze(-1)

            x_0 = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0,
                masked_edges,
            )
            confidence = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0_p, 
                -torch.inf
            )

            transfer_index = torch.zeros_like(x_0, dtype=torch.bool, device=x_0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=transfer_tokens[j, i].item(), largest=True)
                transfer_index[j, select_index] = True

            masked_edges[transfer_index] = x_0[transfer_index]
            masked_edges = rearrange(masked_edges, 'b (h w) -> b h w', h=image_h, w=image_w)

            # Save intermediate prediction (clone to avoid inplace modification)
            intermediate_masked_edges.append(masked_edges.clone().cpu().float().numpy())
            intermeidate_transfer_index.append(transfer_index.cpu().numpy())
            intermediate_predict_confidence.append(x_0_p.cpu().numpy())

        pred_edges = masked_edges.cpu().float().numpy()

        self.maybe_free_model_hooks()

        return {
            "edges": pred_edges,
            "intermediate_masked_edges": intermediate_masked_edges,
            "intermeidate_transfer_index": intermeidate_transfer_index,
            "intermediate_predict_confidence": intermediate_predict_confidence,
        }

class MEMOEdgeRandomDenoiseWithIntermediatePipeline(MEMOEdgeWithIntermediatesPipeline):
    @torch.no_grad()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        temperature: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        denoise_mode: str = 'uniform',
        denoise_kwargs: Optional[Dict[str, Union[int, float]]] = None,
    ) -> Union[Dict, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        image_cond = images.to(self.device)
        image_h, image_w = image_cond.shape[-2:]
        mask_token_id = self.denoiser.config.mask_token_id

        masked_edges = torch.ones(
            (batch_size, image_h, image_w),
            dtype=torch.long,
            device=self._execution_device,
        ) * mask_token_id

        mask_index = (masked_edges == mask_token_id)
        transfer_tokens = self.get_num_transfer_tokens(mask_index, num_inference_steps, denoise_mode=denoise_mode, denoise_kwargs=denoise_kwargs)
        num_masks = mask_index.sum(dim=[1, 2]).view(-1, 1)  # (B, 1)
        mask_ratio_list = torch.flip(transfer_tokens, dims=(1, )).float().cumsum(dim=1) / num_masks.float()  # (B, steps)
        mask_ratio_list = torch.flip(mask_ratio_list, dims=(1, )) # (steps, B), should be 1 -> 0
        # print(f"mask_ratio_list: {mask_ratio_list}")

        intermediate_masked_edges = []
        intermeidate_transfer_index = []
        intermediate_predict_confidence = []

        for i in self.progress_bar(range(num_inference_steps)):
            mask_index = (masked_edges == mask_token_id)
            mask_ratio = torch.tensor(mask_ratio_list[:, i], device=self.device)
            # mask_ratio = mask_ratio.expand(image_cond.shape[0])

            if guidance_scale > 1.0:
                masked_edges_input = torch.cat([masked_edges, masked_edges], dim=0)
                image_cond_input = torch.cat([
                    image_cond, torch.zeros_like(image_cond)
                ], dim=0)
                mask_ratio_input = torch.cat([mask_ratio, mask_ratio], dim=0)
            else:
                masked_edges_input = masked_edges
                image_cond_input = image_cond
                mask_ratio_input = mask_ratio

            edge_logits = self.denoiser(
                masked_edges_input,
                mask_ratio_input,
                image_cond=image_cond_input,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                edge_logits_cond, edge_logits_uncond = edge_logits.chunk(2, dim=0)
                edge_logits = edge_logits_uncond + guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = add_gumbel_noise(edge_logits, temperature=temperature)  # add Gumbel noise for sampling

            edge_logits = rearrange(edge_logits, 'b c h w -> b (h w) c')
            masked_edges = rearrange(masked_edges, 'b h w -> b (h w)')

            x_0 = torch.argmax(edge_logits, dim=-1)
            p = F.softmax(edge_logits, dim=-1)
            x_0_p = torch.gather(p, -1, x_0.unsqueeze(-1)).squeeze(-1)

            x_0 = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0,
                masked_edges,
            )
            confidence = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0_p, 
                -torch.inf
            )

            transfer_index = torch.zeros_like(x_0, dtype=torch.bool, device=x_0.device)
            for j in range(confidence.shape[0]):
                # random select index from index who is not masked
                unmasked_index = torch.nonzero(masked_edges[j] == mask_token_id).squeeze(-1)
                rnd_idx = torch.randperm(unmasked_index.shape[0])
                select_index = unmasked_index[rnd_idx[:transfer_tokens[j, i].item()]]
                transfer_index[j, select_index] = True

            masked_edges[transfer_index] = x_0[transfer_index]
            masked_edges = rearrange(masked_edges, 'b (h w) -> b h w', h=image_h, w=image_w)

            # Save intermediate prediction (clone to avoid inplace modification)
            intermediate_masked_edges.append(masked_edges.clone().cpu().float().numpy())
            intermeidate_transfer_index.append(transfer_index.cpu().numpy())
            intermediate_predict_confidence.append(x_0_p.cpu().numpy())

        pred_edges = masked_edges.cpu().float().numpy()

        self.maybe_free_model_hooks()

        return {
            "edges": pred_edges,
            "intermediate_masked_edges": intermediate_masked_edges,
            "intermeidate_transfer_index": intermeidate_transfer_index,
            "intermediate_predict_confidence": intermediate_predict_confidence,
        }

class MEMOEdgeIntermediateAutoTransferTokenPipeline(MEMOEdgeWithIntermediatesPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        temperature: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        denoise_conf_thres: float = 0.98,
    ) -> Union[Dict, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        image_cond = images.to(self.device)
        image_h, image_w = image_cond.shape[-2:]
        mask_token_id = self.denoiser.config.mask_token_id

        masked_edges = torch.ones(
            (batch_size, image_h, image_w),
            dtype=torch.long,
            device=self._execution_device,
        ) * mask_token_id

        intermediate_masked_edges = []
        intermeidate_transfer_index = []
        intermediate_predict_confidence = []

        for i in self.progress_bar(range(max_inference_steps)):
            mask_index = (masked_edges == mask_token_id)
            if mask_index.sum() == 0:
                break
            mask_ratio = mask_index.sum(dim=[1, 2]).float() / (image_h * image_w)  # (B, )
            # print(mask_ratio)

            if guidance_scale > 1.0:
                masked_edges_input = torch.cat([masked_edges, masked_edges], dim=0)
                image_cond_input = torch.cat([
                    image_cond, torch.zeros_like(image_cond)
                ], dim=0)
                mask_ratio_input = torch.cat([mask_ratio, mask_ratio], dim=0)
            else:
                masked_edges_input = masked_edges
                image_cond_input = image_cond
                mask_ratio_input = mask_ratio

            edge_logits = self.denoiser(
                masked_edges_input,
                mask_ratio_input,
                image_cond=image_cond_input,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                edge_logits_cond, edge_logits_uncond = edge_logits.chunk(2, dim=0)
                edge_logits = edge_logits_uncond + guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = add_gumbel_noise(edge_logits, temperature=temperature)  # add Gumbel noise for sampling

            edge_logits = rearrange(edge_logits, 'b c h w -> b (h w) c')
            masked_edges = rearrange(masked_edges, 'b h w -> b (h w)')

            x_0 = torch.argmax(edge_logits, dim=-1)
            p = F.softmax(edge_logits, dim=-1)
            x_0_p = torch.gather(p, -1, x_0.unsqueeze(-1)).squeeze(-1)

            x_0 = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0,
                masked_edges,
            )
            confidence = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0_p, 
                -torch.inf
            )

            transfer_index = torch.zeros_like(x_0, dtype=torch.bool, device=x_0.device)
            for j in range(confidence.shape[0]):
                this_masked_tokens = (masked_edges[j] == mask_token_id).sum().item()
                if i < max_inference_steps - 1:
                    num_transfer = max((confidence[j] > denoise_conf_thres).sum().item(), int(np.ceil(0.3 * this_masked_tokens)))
                    num_transfer = min(num_transfer, int(0.2 * image_h * image_w))
                else:
                    # in the last step, we transfer all masked tokens
                    num_transfer = this_masked_tokens
                _, select_index = torch.topk(confidence[j], k=num_transfer, largest=True)
                transfer_index[j, select_index] = True

            masked_edges[transfer_index] = x_0[transfer_index]
            masked_edges = rearrange(masked_edges, 'b (h w) -> b h w', h=image_h, w=image_w)

            # Save intermediate prediction (clone to avoid inplace modification)
            intermediate_masked_edges.append(masked_edges.clone().cpu().float().numpy())
            intermeidate_transfer_index.append(transfer_index.cpu().numpy())
            intermediate_predict_confidence.append(x_0_p.cpu().numpy())

        pred_edges = masked_edges.cpu().float().numpy()

        self.maybe_free_model_hooks()

        return {
            "edges": pred_edges,
            "intermediate_masked_edges": intermediate_masked_edges,
            "intermeidate_transfer_index": intermeidate_transfer_index,
            "intermediate_predict_confidence": intermediate_predict_confidence,
        }

def local_maxima_map(img: torch.Tensor, connectivity: int = 8) -> torch.Tensor:
    """
    Args:
        img: 2D tensor (H, W)
        connectivity: 4 or 8

    Returns:
        2D bool tensor (H, W) with True where the pixel is a local maximum
    """
    assert img.ndim == 2, "Image must be 2D"
    assert connectivity in (4, 8), "Connectivity must be 4 or 8"

    H, W = img.shape
    neighbors = []

    # 4-connectivity: up, down, left, right
    neighbors.append(torch.roll(img, shifts=1, dims=0))   # up
    neighbors.append(torch.roll(img, shifts=-1, dims=0))  # down
    neighbors.append(torch.roll(img, shifts=1, dims=1))   # left
    neighbors.append(torch.roll(img, shifts=-1, dims=1))  # right

    if connectivity == 8:
        neighbors.append(torch.roll(img, shifts=(1, 1), dims=(0, 1)))   # up-left
        neighbors.append(torch.roll(img, shifts=(1, -1), dims=(0, 1)))  # up-right
        neighbors.append(torch.roll(img, shifts=(-1, 1), dims=(0, 1)))  # down-left
        neighbors.append(torch.roll(img, shifts=(-1, -1), dims=(0, 1))) # down-right

    is_max = torch.ones_like(img, dtype=torch.bool)
    for n in neighbors:
        is_max &= img >= n

    return is_max

def local_maxima_map_v2(img: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Args:
        img: 2D tensor (H, W)
        radius: non-negative integer r. Uses a square neighborhood of size (2r+1)x(2r+1)
                (Chebyshev radius). r=0 returns all True.

    Returns:
        2D bool tensor (H, W) with True where the pixel equals the maximum
        within its (2r+1)x(2r+1) neighborhood. Ties (plateaus) are considered maxima.
    """
    assert img.ndim == 2, "Image must be 2D"
    assert isinstance(radius, int) and radius >= 0, "radius must be a non-negative integer"

    if radius == 0:
        return torch.ones_like(img, dtype=torch.bool)

    # Use -inf padding so borders aren't biased. Cast to float for pooling.
    x = img.to(torch.float32)
    k = 2 * radius + 1

    x_pad = F.pad(
        x.unsqueeze(0).unsqueeze(0),  # NCHW
        pad=(radius, radius, radius, radius),
        mode="constant",
        value=float("-inf"),
    )
    max_in_window = F.max_pool2d(x_pad, kernel_size=k, stride=1).squeeze(0).squeeze(0)

    # Plateau-friendly comparison (same as using >= against all neighbors in v1)
    is_max = x == max_in_window
    return is_max

class MEMOEdgeIntermediateLocalMaximumPipeline(MEMOEdgeWithIntermediatesPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        temperature: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        denoise_conf_thres: float = 0.98,
    ) -> Union[Dict, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        image_cond = images.to(self.device)
        image_h, image_w = image_cond.shape[-2:]
        mask_token_id = self.denoiser.config.mask_token_id

        masked_edges = torch.ones(
            (batch_size, image_h, image_w),
            dtype=torch.long,
            device=self._execution_device,
        ) * mask_token_id

        intermediate_masked_edges = []
        intermeidate_transfer_index = []
        intermediate_predict_confidence = []

        for i in self.progress_bar(range(max_inference_steps)):
            mask_index = (masked_edges == mask_token_id)
            if mask_index.sum() == 0:
                break
            mask_ratio = mask_index.sum(dim=[1, 2]).float() / (image_h * image_w)  # (B, )
            # print(mask_ratio)

            if guidance_scale > 1.0:
                masked_edges_input = torch.cat([masked_edges, masked_edges], dim=0)
                image_cond_input = torch.cat([
                    image_cond, torch.zeros_like(image_cond)
                ], dim=0)
                mask_ratio_input = torch.cat([mask_ratio, mask_ratio], dim=0)
            else:
                masked_edges_input = masked_edges
                image_cond_input = image_cond
                mask_ratio_input = mask_ratio

            edge_logits = self.denoiser(
                masked_edges_input,
                mask_ratio_input,
                image_cond=image_cond_input,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                edge_logits_cond, edge_logits_uncond = edge_logits.chunk(2, dim=0)
                edge_logits = edge_logits_uncond + guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = add_gumbel_noise(edge_logits, temperature=temperature)  # add Gumbel noise for sampling

            edge_logits = rearrange(edge_logits, 'b c h w -> b (h w) c')
            masked_edges = rearrange(masked_edges, 'b h w -> b (h w)')

            x_0 = torch.argmax(edge_logits, dim=-1)
            p = F.softmax(edge_logits, dim=-1)
            x_0_p = torch.gather(p, -1, x_0.unsqueeze(-1)).squeeze(-1)

            x_0 = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0,
                masked_edges,
            )
            confidence = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0_p, 
                -torch.inf
            )

            transfer_index = torch.zeros_like(x_0, dtype=torch.bool, device=x_0.device)
            for j in range(confidence.shape[0]):
                # local_max_map = local_maxima_map(
                #     rearrange(confidence[j], '(h w) -> h w', h=image_h, w=image_w),
                #     connectivity=8
                # )  # (H, W)
                local_max_map = local_maxima_map_v2(
                    rearrange(confidence[j], '(h w) -> h w', h=image_h, w=image_w),
                    radius=1
                )  # (H, W)
                local_max_map = rearrange(local_max_map, 'h w -> (h w)')
                unmasked_index = torch.nonzero(
                    (masked_edges[j] == mask_token_id) & local_max_map
                ).squeeze(-1) # being both masked and local maxima
                rnd_idx = torch.randperm(unmasked_index.shape[0])

                # decided number of tokens to transfer
                if i < max_inference_steps - 1:
                    num_transfer = min(
                        len(unmasked_index), 
                        int(0.2 * image_h * image_w)
                    )
                    select_index = unmasked_index[rnd_idx[:num_transfer]]
                else:
                    select_index = torch.nonzero(masked_edges[j] == mask_token_id).squeeze(-1)  # transfer all masked tokens

                # this_masked_tokens = (masked_edges[j] == mask_token_id).sum().item()
                # _, select_index = torch.topk(confidence[j], k=num_transfer, largest=True)
                transfer_index[j, select_index] = True

            masked_edges[transfer_index] = x_0[transfer_index]
            masked_edges = rearrange(masked_edges, 'b (h w) -> b h w', h=image_h, w=image_w)

            # Save intermediate prediction (clone to avoid inplace modification)
            intermediate_masked_edges.append(masked_edges.clone().cpu().float().numpy())
            intermeidate_transfer_index.append(transfer_index.cpu().numpy())
            intermediate_predict_confidence.append(x_0_p.cpu().numpy())

        pred_edges = masked_edges.cpu().float().numpy()

        self.maybe_free_model_hooks()

        return {
            "edges": pred_edges,
            "intermediate_masked_edges": intermediate_masked_edges,
            "intermeidate_transfer_index": intermeidate_transfer_index,
            "intermediate_predict_confidence": intermediate_predict_confidence,
        }

class MEMOEdgeIntermediateLocalMaximumPipeline(MEMOEdgeWithIntermediatesPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        images: List[Union[np.array, torch.Tensor]],
        temperature: float = 0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        max_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        denoise_conf_thres: float = 0.98,
    ) -> Union[Dict, Tuple]:

        if isinstance(images[0], np.ndarray):
            images = torch.from_numpy(images).to(self._execution_device)
            if images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)

        if images.max() > 1:
            images = images / 255.0

        batch_size = len(images)
        image_cond = images.to(self.device)
        image_h, image_w = image_cond.shape[-2:]
        mask_token_id = self.denoiser.config.mask_token_id

        masked_edges = torch.ones(
            (batch_size, image_h, image_w),
            dtype=torch.long,
            device=self._execution_device,
        ) * mask_token_id

        pred_probs = torch.zeros(
            (batch_size, image_h * image_w, self.denoiser.config.out_channels),
            dtype=torch.float32,
            device=self._execution_device,
        )

        intermediate_masked_edges = []
        intermeidate_transfer_index = []
        intermediate_predict_confidence = []
        allowed_highest_confidence = None

        for i in self.progress_bar(range(max_inference_steps)):
            mask_index = (masked_edges == mask_token_id)
            if mask_index.sum() == 0:
                break
            mask_ratio = mask_index.sum(dim=[1, 2]).float() / (image_h * image_w)  # (B, )
            # print(mask_ratio)

            if guidance_scale > 1.0:
                masked_edges_input = torch.cat([masked_edges, masked_edges], dim=0)
                image_cond_input = torch.cat([
                    image_cond, torch.zeros_like(image_cond)
                ], dim=0)
                mask_ratio_input = torch.cat([mask_ratio, mask_ratio], dim=0)
            else:
                masked_edges_input = masked_edges
                image_cond_input = image_cond
                mask_ratio_input = mask_ratio

            edge_logits = self.denoiser(
                masked_edges_input,
                mask_ratio_input,
                image_cond=image_cond_input,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                edge_logits_cond, edge_logits_uncond = edge_logits.chunk(2, dim=0)
                edge_logits = edge_logits_uncond + guidance_scale * (edge_logits_cond - edge_logits_uncond)

            edge_logits = add_gumbel_noise(edge_logits, temperature=temperature)  # add Gumbel noise for sampling

            edge_logits = rearrange(edge_logits, 'b c h w -> b (h w) c')
            masked_edges = rearrange(masked_edges, 'b h w -> b (h w)')

            x_0 = torch.argmax(edge_logits, dim=-1)
            p = F.softmax(edge_logits, dim=-1)
            x_0_p = torch.gather(p, -1, x_0.unsqueeze(-1)).squeeze(-1)

            x_0 = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0,
                masked_edges,
            )
            confidence = torch.where(
                rearrange(mask_index, 'b h w -> b (h w)'),
                x_0_p, 
                -torch.inf
            )
            if allowed_highest_confidence is None:
                allowed_highest_confidence = confidence
            else:
                # To ensure that the confidence of newly transferred tokens is strictly decreasing than its previous ones
                allowed_highest_confidence = torch.min(allowed_highest_confidence, confidence)

            transfer_index = torch.zeros_like(x_0, dtype=torch.bool, device=x_0.device)
            for j in range(confidence.shape[0]):
                # local_max_map = local_maxima_map(
                #     rearrange(confidence[j], '(h w) -> h w', h=image_h, w=image_w),
                #     connectivity=8
                # )  # (H, W)
                local_max_map = local_maxima_map_v2(
                    rearrange(confidence[j], '(h w) -> h w', h=image_h, w=image_w),
                    radius=1
                )  # (H, W)
                local_max_map = rearrange(local_max_map, 'h w -> (h w)')
                unmasked_index = torch.nonzero(
                    (masked_edges[j] == mask_token_id) & local_max_map
                ).squeeze(-1) # being both masked and local maxima
                rnd_idx = torch.randperm(unmasked_index.shape[0])

                # decided number of tokens to transfer
                if i < max_inference_steps - 1:
                    num_transfer = min(
                        len(unmasked_index), 
                        int(0.2 * image_h * image_w)
                    )
                    select_index = unmasked_index[rnd_idx[:num_transfer]]
                else:
                    select_index = torch.nonzero(masked_edges[j] == mask_token_id).squeeze(-1)  # transfer all masked tokens

                # this_masked_tokens = (masked_edges[j] == mask_token_id).sum().item()
                # _, select_index = torch.topk(confidence[j], k=num_transfer, largest=True)
                transfer_index[j, select_index] = True

            masked_edges[transfer_index] = x_0[transfer_index]
            masked_edges = rearrange(masked_edges, 'b (h w) -> b h w', h=image_h, w=image_w)
            pred_probs[transfer_index] = p[transfer_index]

            # Save intermediate prediction (clone to avoid inplace modification)
            intermediate_masked_edges.append(masked_edges.clone().cpu().float().numpy())
            intermeidate_transfer_index.append(transfer_index.cpu().numpy())
            intermediate_predict_confidence.append(x_0_p.cpu().numpy())

        pred_edges = masked_edges.cpu().float().numpy()
        pred_probs = rearrange(pred_probs, 'b (h w) c -> b h w c', h=image_h, w=image_w)  # (B, H, W, C)
        pred_probs = pred_probs.cpu().float().numpy()  # (B, H*W, C)

        self.maybe_free_model_hooks()

        return {
            "edges": pred_edges,
            "intermediate_masked_edges": intermediate_masked_edges,
            "intermeidate_transfer_index": intermeidate_transfer_index,
            "intermediate_predict_confidence": intermediate_predict_confidence,
            "pred_probs": pred_probs,
        }
