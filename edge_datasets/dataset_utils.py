import torch
from torch.utils.data import Dataset
from misc_utils.model_utils import instantiate_from_config
import warnings
import numpy as np
import albumentations as A
from einops import rearrange, repeat
from .random_mask_utils import RandomIrregularMaskEmbedder

# Suppress specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

stacked_tensor_batching = lambda batch, key: torch.stack([torch.tensor(b[key]) for b in batch])
tensor_batching = lambda batch, key: [torch.tensor(b[key]) for b in batch]
list_batching = lambda batch, key : [b[key] for b in batch]

def bsds_full_resolution_image_edge_collate_fn(batch):
    def random_rotate_image_and_edge_90_degrees(image, edge):
        rnd_num = np.random.rand()
        if rnd_num < 0.5:
            # rotate 90 degrees clockwise
            rot_image = np.rot90(image, k=3)
            rot_edge = np.rot90(edge, k=3)
        else:
            # rotate 90 degrees counter-clockwise
            rot_image = np.rot90(image, k=1)
            rot_edge = np.rot90(edge, k=1)
        return np.array(rot_image), np.array(rot_edge)

    images = [b['image'] for b in batch]
    edges = [b['edge'] for b in batch]
    post_rotate_images = []
    post_rotate_edges = []
    rnd_num = np.random.rand()
    if rnd_num < 0.5:
        # set to 480x320
        for img, edg in zip(images, edges):
            h, w = img.shape[:2]
            if h > w:
                post_rotate_images.append(img)
                post_rotate_edges.append(edg)
            else:
                rot_img, rot_edg = random_rotate_image_and_edge_90_degrees(img, edg)
                post_rotate_images.append(rot_img)
                post_rotate_edges.append(rot_edg)
    else:
        # set to 320x480
        for img, edg in zip(images, edges):
            h, w = img.shape[:2]
            if h > w:
                rot_img, rot_edg = random_rotate_image_and_edge_90_degrees(img, edg)
                post_rotate_images.append(rot_img)
                post_rotate_edges.append(rot_edg)
            else:
                post_rotate_images.append(img)
                post_rotate_edges.append(edg)

    edges = torch.stack([torch.tensor(edg) for edg in post_rotate_edges])
    images = torch.stack([torch.tensor(img) for img in post_rotate_images])
    images = rearrange(images, 'b h w c -> b c h w')
    
    res = {
        'image_path': list_batching(batch, 'image_path'),
        'edge_path': list_batching(batch, 'edge_path'),
        'image': images,
        'edge': edges,
    }
    return res
        

def edge_collate_fn(batch):
    # the edge will be the training target, so we rename it to 'image'
    edges = stacked_tensor_batching(batch, 'edge')

    # we repeat the channels to match the input image
    edges = repeat(edges, 'b h w -> b c h w', c=3)

    res = {
        'image_path': list_batching(batch, 'edge_path'),
        'image': edges,
    }
    return res

def edge_color_collate_fn(batch):
    # the edge will be the training target, so we rename it to 'image'
    edges = stacked_tensor_batching(batch, 'edge')

    edges = rearrange(edges, 'b h w c -> b c h w')

    res = {
        'image_path': list_batching(batch, 'edge_path'),
        'image': edges,
    }
    return res

def image_edge_unet_collate_fn(batch):
    # image: b, h, w, c -> b, c, h, w
    # edge: b, h, w -> b, h, w
    edges = stacked_tensor_batching(batch, 'edge')

    images = stacked_tensor_batching(batch, 'image')
    images = rearrange(images, 'b h w c -> b c h w')

    res = {
        'image_path': list_batching(batch, 'image_path'),
        'edge_path': list_batching(batch, 'edge_path'),
        'image': images,
        'edge': edges,
    }

    if 'edge_labels' in batch[0]:
        edge_labels = stacked_tensor_batching(batch, 'edge_labels') # b, h, w
        res['edge_labels'] = edge_labels

    if 'edge_index' in batch[0]:
        edge_labels = stacked_tensor_batching(batch, 'edge_index') # b, h, w
        res['edge_index'] = edge_labels

    if 'junction_uncertainty' in batch[0]:
        junction_uncertainty = stacked_tensor_batching(batch, 'junction_uncertainty')
        res['junction_uncertainty'] = junction_uncertainty # b, h, w

    if 'edge_mask' in batch[0]:
        edge_mask = stacked_tensor_batching(batch, 'edge_mask')
        res['edge_mask'] = edge_mask # b, h, w

    return res

def image_edge_collate_fn(batch):
    edges = stacked_tensor_batching(batch, 'edge')

    # we repeat the channels to match the input image
    edges = repeat(edges, 'b h w -> b c h w', c=3)

    images = stacked_tensor_batching(batch, 'image')
    images = rearrange(images, 'b h w c -> b c h w')

    res = {
        'image_path': list_batching(batch, 'image_path'),
        'edge_path': list_batching(batch, 'edge_path'),
        'image': images,
        'edge': edges,
    }
    return res

def image_edge_color_collate_fn(batch):
    edges = stacked_tensor_batching(batch, 'edge')
    edges = rearrange(edges, 'b h w c -> b c h w')

    images = stacked_tensor_batching(batch, 'image')
    images = rearrange(images, 'b h w c -> b c h w')

    res = {
        'image_path': list_batching(batch, 'image_path'),
        'edge_path': list_batching(batch, 'edge_path'),
        'image': images,
        'edge': edges,
    }
    return res

def adaptive_mask_edge_collate_fn(batch):
    image_edge_res = image_edge_collate_fn(batch)

    masks = list_batching(batch, 'masks')
    thres_mask = stacked_tensor_batching(batch, 'thres_mask')

    thresed_edge = stacked_tensor_batching(batch, 'thresed_edge')
    thresed_edge = repeat(thresed_edge, 'b h w -> b c h w', c=3)

    image_edge_res['masks'] = masks
    image_edge_res['thres_mask'] = thres_mask
    image_edge_res['thresed_edge'] = thresed_edge

    return image_edge_res

'''
Difference between JointDataset and ConcatDataset
JointDataset: samples from multiple datasets with different sampling rates, but it cannot ensure the same index will always return the same sample
ConcatDataset: simple concatenation of datasets without sampling rates, it ensures the same index will always return the same sample
'''

class JointDataset(Dataset):
    def __init__(self, subdatasets, sampling_rates=None, mode='train'):
        self.datasets = []
        for dataset_config in subdatasets:
            dataset = instantiate_from_config(dataset_config)
            self.datasets.append(dataset)
        self.sampling_rates = sampling_rates if sampling_rates is not None else [1.] * len(self.datasets)

        # Calculate total number of samples and sampling weights
        # self.total_samples = int(sum([len(d) * r for d, r in zip(self.datasets, self.sampling_rates)]))
        # self.weights = [len(d) * r / self.total_samples for d, r in zip(self.datasets, self.sampling_rates)]
        self.total_samples = int(sum([len(d) for d in self.datasets]))
        self.weights = [r for r in self.sampling_rates]

        self.mode = mode
        print('INFO: total number of samples in JointDataset is {}'.format(self.total_samples))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which dataset to sample from
        dataset_idx = torch.multinomial(torch.tensor(self.weights), 1).item()
        dataset = self.datasets[dataset_idx]

        # Sample from the chosen dataset
        sample_idx = torch.randint(len(dataset), (1,)).item() # if self.mode == 'train' else idx % len(dataset)
        return dataset[sample_idx]

class ConcatDataset(Dataset):
    # simple concatenation of datasets without sampling rates
    def __init__(self, subdatasets, mode='train'):
        self.datasets = []
        for dataset_config in subdatasets:
            dataset = instantiate_from_config(dataset_config)
            self.datasets.append(dataset)

        # Calculate total number of samples and sampling weights
        self.total_samples = int(sum([len(d) for d in self.datasets]))
        self.sample_cumsum = np.cumsum([0] + [len(d) for d in self.datasets])

        self.mode = mode
        print('INFO: total number of samples in ConcatDataset is {}'.format(self.total_samples))

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Determine which dataset to sample from
        dataset_idx = np.argmax(idx < self.sample_cumsum) - 1
        dataset = self.datasets[dataset_idx]

        # Sample from the chosen dataset
        sample_idx = idx - self.sample_cumsum[dataset_idx]
        return dataset[sample_idx]

class JointDatasetWithRndMask(JointDataset):
    def __init__(
            self, 
            *args, 
            irregular_kwargs = {
                'max_angle': 4, 
                'max_len': 60, 
                'max_width': 40, 
                'min_len': 50, 
                'min_width': 20, 
                'min_times': 2, 
                'max_times': 8
            },
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rnd_mask_generator = RandomIrregularMaskEmbedder(**irregular_kwargs)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        edge = batch['edge']
        h, w = edge.shape
        edge_mask = self.rnd_mask_generator((1, h, w))[0] # (h, w)
        batch['edge_mask'] = edge_mask
        return batch

class ConcatDatasetWithRndMask(ConcatDataset):
    def __init__(
            self, 
            *args, 
            irregular_kwargs = {
                'max_angle': 4, 
                'max_len': 60, 
                'max_width': 40, 
                'min_len': 50, 
                'min_width': 20, 
                'min_times': 2, 
                'max_times': 8
            },
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rnd_mask_generator = RandomIrregularMaskEmbedder(**irregular_kwargs)

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        edge = batch['edge']
        h, w = edge.shape
        edge_mask = self.rnd_mask_generator((1, h, w))[0] # (h, w)
        batch['edge_mask'] = edge_mask
        return batch