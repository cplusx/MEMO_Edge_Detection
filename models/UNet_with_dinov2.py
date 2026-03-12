import torch
from torch import nn
from diffusers.models.unets.unet_2d import UNet2DModel, UNet2DOutput

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def preprocess_image(x):
    mean = MEAN.to(device=x.device, dtype=x.dtype)
    std = STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std

class UNet2DwithDINOv2(UNet2DModel):
    def __init__(self, *args, cond_dim=3, dino_size=(224, 224), dino_name='dinov2_vitb14_reg', **kwargs):
        super().__init__(*args, **kwargs)

        in_channels = self.conv_in.in_channels
        out_channels = self.conv_out.out_channels
        self.embed_num = out_channels + 1 # + 1 for the [mask] token
        self.mask_token_id = out_channels  # The last token is the mask token
        self.register_to_config(mask_token_id=self.mask_token_id)
        self.embedder = nn.Embedding(self.embed_num, in_channels)

        self.pre_conv = nn.Conv2d(
            in_channels + cond_dim, # +3 for RGB channels
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )

        self.dino = torch.hub.load(
            'facebookresearch/dinov2',
            dino_name,
        )

        self.dino_to_unet_feat_map = nn.Conv2d(
            self.dino.embed_dim,
            self.block_out_channels[-1],
            kernel_size=1,
        )
        self.dino_size = tuple(dino_size)

    def get_dino_features(self, sample, dino_size=None):
        if dino_size is None:
            dino_size = self.dino_size
        resized_sample = nn.functional.interpolate(
            sample, 
            size=dino_size,
            mode='bilinear', 
            align_corners=False
        )
        dino_input = preprocess_image(resized_sample)
        dino_features = self.dino.get_intermediate_layers(dino_input, reshape=True)[0]
        dino_features = self.dino_to_unet_feat_map(dino_features)
        return dino_features

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        image_cond: torch.Tensor,
        class_labels: torch.Tensor = None,
        return_dict: bool = True,
        dino_features: torch.Tensor = None,
    ):
        sample = self.embedder(sample.long())
        sample = sample.permute(0, 3, 1, 2)  # Convert to (batch_size, channels, height, width)

        sample = torch.cat(
            [sample, image_cond], dim=1
        )
        sample = self.pre_conv(sample)

        return self.forward_unet_with_dino(
            sample=sample,
            timestep=timestep,
            image_cond=image_cond,
            class_labels=class_labels,
            return_dict=return_dict,
            dino_features=dino_features,
        )

    def forward_unet_with_dino(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        image_cond: torch.Tensor,
        class_labels: torch.Tensor = None,
        return_dict: bool = True,
        dino_features: torch.Tensor = None,
    ):
        if dino_features is None:
            dino_features = self.get_dino_features(image_cond)

        # copy the original forward in UNet2DModel
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)
        h, w = sample.shape[2], sample.shape[3]
        dino_features = nn.functional.interpolate(
            dino_features,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        sample = sample + dino_features  # Add DINO features to the mid block output

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)
