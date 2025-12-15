import os
import glob
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
import cv2
from .image_augmentor import ImageAugmentor
import scipy.sparse as sp

IMAGE_DIR = 'edge_data/BSDS500'
GRANULAR_EDGE_DIR = 'edge_data/repurposed_BSDS_edges'
DATASET_META = {
    'train': {
        'images_dir': '{}/BSDS500/data/images/train',
        'edge_dir': '{}/train/granular_edges'
    },
    'val': {
        'images_dir': '{}/BSDS500/data/images/val/',
        'edge_dir': '{}/val/granular_edges'
    },
    'train_val': {
        'images_dir': '{}/BSDS500/data/images/train_val',
        'edge_dir': '{}/train_val/granular_edges'
    },
    'test': {
        'images_dir': '{}/BSDS500/data/images/test/',
        'edge_dir': '{}/test/granular_edges'
    }
}

class GranularBSDSDataset(Dataset):
    def __init__(
        self, 
        image_dir, 
        granular_edge_dir,
        mode='train', 
        max_samples=-1,
        image_shape=(256, 256), 
        size_range=(1.0, 1.0),
        aspect_ratio_range=(1.0, 1.0), 
        perspective_range=0.0, 
        rotation_range=0,
        vertical_flip=True,
        horizontal_flip=True,
        transpose_flip=True,
        max_granularity=6,
    ):
        self.image_dir = image_dir
        self.graular_edge_dir = granular_edge_dir
        self.mode = mode
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.transpose_flip = transpose_flip
        self.max_granularity = max_granularity

        # Build the image and edge map paths
        images_dir = DATASET_META[mode]['images_dir'].format(self.image_dir)
        edge_dir = DATASET_META[mode]['edge_dir'].format(self.graular_edge_dir)

        # Get list of image files
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '**/*.jpg'), recursive=True))
        self.edge_files = sorted(glob.glob(os.path.join(edge_dir, '**/*.png'), recursive=True))

        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]
            self.edge_files = self.edge_files[:max_samples]

        assert len(self.image_files) == len(self.edge_files), "Number of images and edge maps must be equal."

        # Create an instance of the augmentation handler
        self.augmentor = ImageAugmentor(
            image_shape=image_shape,
            size_range=size_range,
            aspect_ratio_range=aspect_ratio_range,
            perspective_range=perspective_range,
            rotation_range=rotation_range,
            vertical_flip=vertical_flip,
            horizontal_flip=horizontal_flip
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # return self.get_by_idx(idx)
        try:
            return self.get_by_idx(idx)
        except Exception as e:
            image_path = self.image_files[idx]
            print(f" {__file__}: Error loading image {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')

        if self.augmentor is not None:
            # Pad images to square before augmentation
            image, edge = self.augmentor.pad_to_square(image, edge)

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge = np.where(edge > self.max_granularity, self.max_granularity, edge)

            h, w = self.image_shape
            H, W = image.shape[:2]

            if H < h or W < w:
                raise ValueError(f"Image size {image.shape} is smaller than target size {self.image_shape}")
            x1, y1 = np.random.randint(0, W - w + 1), np.random.randint(0, H - h + 1)
            x2, y2 = x1 + w, y1 + h
            
            image = image[y1:y2, x1:x2]
            edge = edge[y1:y2, x1:x2]

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.transpose(image, (1, 0, 2))
                    edge = np.transpose(edge, (1, 0))
            if self.horizontal_flip and np.random.rand() > 0.5:
                image = np.fliplr(image)
                edge = np.fliplr(edge)
            if self.vertical_flip and np.random.rand() > 0.5:
                image = np.flipud(image)
                edge = np.flipud(edge)

        else:

            image = np.array(image) / 255.0
            edge = np.array(edge)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image,
            'edge': edge
        }

class EndlessGranularBSDSDataset(GranularBSDSDataset):

    def __len__(self):
        return len(self.image_files) * 1000

    def __getitem__(self, idx):
        return super().__getitem__(idx % len(self.image_files))

        
class GranularBSDSDatasetWithEdgeLabels(GranularBSDSDataset):
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]
        edge_label_path = edge_path.replace('/granular_edges/', '/granular_edge_components/').replace('.png', '.npz')

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')
        edge_labels = sp.load_npz(edge_label_path).toarray()

        num_components = edge_labels.max()
        if num_components > 255:
            # pick components with largest area
            component_area = np.bincount(edge_labels.flatten())
            component_area[0] = 0
            sorted_area = np.argsort(component_area)[::-1]
            mask = np.isin(edge_labels, sorted_area[:255], invert=True)
            edge_labels[mask] = 0
            # map index to 0-255
            unique_labels = np.unique(edge_labels)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            edge_labels = np.vectorize(label_mapping.get)(edge_labels).astype(np.uint8)

        edge_labels = Image.fromarray(edge_labels).convert('L')

        if self.augmentor is not None:
            # Pad images to square before augmentation
            _, edge_labels = self.augmentor.pad_to_square(image, edge_labels)
            image, edge = self.augmentor.pad_to_square(image, edge)

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge = np.where(edge > self.max_granularity, self.max_granularity, edge)
            edge_labels = np.array(edge_labels)

            h, w = self.image_shape
            H, W = image.shape[:2]

            if H < h or W < w:
                raise ValueError(f"Image size {image.shape} is smaller than target size {self.image_shape}")
            x1, y1 = np.random.randint(0, W - w + 1), np.random.randint(0, H - h + 1)
            x2, y2 = x1 + w, y1 + h
            
            image = image[y1:y2, x1:x2]
            edge = edge[y1:y2, x1:x2]
            edge_labels = edge_labels[y1:y2, x1:x2]

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.transpose(image, (1, 0, 2))
                    edge = np.transpose(edge, (1, 0))
                    edge_labels = np.transpose(edge_labels, (1, 0))
            if self.horizontal_flip and np.random.rand() > 0.5:
                image = np.fliplr(image)
                edge = np.fliplr(edge)
                edge_labels = np.fliplr(edge_labels)
            if self.vertical_flip and np.random.rand() > 0.5:
                image = np.flipud(image)
                edge = np.flipud(edge)
                edge_labels = np.flipud(edge_labels)

        else:

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge_labels = np.array(edge_labels)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)
        edge_labels = edge_labels.astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image,
            'edge': edge,
            'edge_labels': edge_labels
        }

class EndlessGranularBSDSDatasetWithEdgeLabels(GranularBSDSDatasetWithEdgeLabels):
    def __len__(self):
        return len(self.image_files) * 1000

    def __getitem__(self, idx):
        return super().__getitem__(idx % len(self.image_files))

class GranularBSDSDatasetWithEdgeLabelsNoAugmentation(GranularBSDSDatasetWithEdgeLabels):
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]
        edge_label_path = edge_path.replace('/granular_edges/', '/granular_edge_components/').replace('.png', '.npz')

        image = Image.open(image_path).convert('RGB')
        edge = Image.open(edge_path).convert('L')
        edge_labels = sp.load_npz(edge_label_path).toarray()

        num_components = edge_labels.max()
        if num_components > 255:
            # pick components with largest area
            component_area = np.bincount(edge_labels.flatten())
            component_area[0] = 0
            sorted_area = np.argsort(component_area)[::-1]
            mask = np.isin(edge_labels, sorted_area[:255], invert=True)
            edge_labels[mask] = 0
            # map index to 0-255
            unique_labels = np.unique(edge_labels)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            edge_labels = np.vectorize(label_mapping.get)(edge_labels).astype(np.uint8)

        edge_labels = Image.fromarray(edge_labels).convert('L')

        if self.augmentor is not None:

            image = np.array(image)[1:, 1:] / 255.0
            edge = np.array(edge)[1:, 1:]
            edge = np.where(edge > self.max_granularity, self.max_granularity, edge)
            edge_labels = np.array(edge_labels)[1:, 1:]

            if self.horizontal_flip and np.random.rand() > 0.5:
                image = np.fliplr(image)
                edge = np.fliplr(edge)
                edge_labels = np.fliplr(edge_labels)

        else:

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge_labels = np.array(edge_labels)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)
        edge_labels = edge_labels.astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image,
            'edge': edge,
            'edge_labels': edge_labels
        }

class EndlessGranularBSDSDatasetWithEdgeLabelsNoAugmentation(GranularBSDSDatasetWithEdgeLabelsNoAugmentation):
    def __len__(self):
        return len(self.image_files) * 1000

    def __getitem__(self, idx):
        return super().__getitem__(idx % len(self.image_files))

class BinaryRepursedBSDSDataset(GranularBSDSDataset):
    def get_by_idx(self, idx):
        res = super().get_by_idx(idx)
        res['edge'] = np.where(res['edge'] > 0, 1, 0).astype(np.uint8)
        return res

class EndlessBinaryRepursedBSDSDataset(BinaryRepursedBSDSDataset):
    def __len__(self):
        return len(self.image_files) * 1000

    def __getitem__(self, idx):
        return super().__getitem__(idx % len(self.image_files))
