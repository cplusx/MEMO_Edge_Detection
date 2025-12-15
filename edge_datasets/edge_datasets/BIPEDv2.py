import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import cv2
from .image_augmentor import ImageAugmentor

DATASET_DIR = 'edge_data/BIPEDv2'
DATASET_META = {
    'train': {
        'images_dir': '{}/BIPEDv2/BIPED/edges/imgs/train/rgbr/real',
        'edge_dir': '{}/BIPEDv2/BIPED/edges/edge_maps/train/rgbr/real'
    },
    'test': {
        'images_dir': '{}/BIPEDv2/BIPED/edges/imgs/test/rgbr',
        'edge_dir': '{}/BIPEDv2/BIPED/edges/edge_maps/test/rgbr'
    }
}

class BIPEDv2Dataset(Dataset):
    def __init__(
        self, 
        root_dir, 
        mode='train', 
        max_samples=-1,
        image_shape=(256, 256), 
        size_range=(1.0, 1.0),
        aspect_ratio_range=(1.0, 1.0), 
        perspective_range=0.0, 
        rotation_range=0,
        vertical_flip=True,
        horizontal_flip=True
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        # Build the image and edge map paths
        images_dir = DATASET_META[mode]['images_dir'].format(root_dir)
        edge_dir = DATASET_META[mode]['edge_dir'].format(root_dir)

        # Get list of image files
        self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        self.edge_files = sorted(glob.glob(os.path.join(edge_dir, '*.png')))

        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]
            self.edge_files = self.edge_files[:max_samples]

        assert len(self.image_files) == len(self.edge_files), "Number of images and edge maps must be equal."

        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection('opencv_edge/model.yml.gz')
        # Create an instance of the augmentation handler
        if mode == 'train':
            self.augmentor = ImageAugmentor(
                image_shape=image_shape,
                size_range=size_range,
                aspect_ratio_range=aspect_ratio_range,
                perspective_range=perspective_range,
                rotation_range=rotation_range,
                vertical_flip=vertical_flip,
                horizontal_flip=horizontal_flip
            )
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
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

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge = self.augmentor.apply(image, edge, transform_params)
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.BILINEAR)

            image = np.array(image) / 255.0
            edge = np.array(edge) / 255.0

        image = image.astype(np.float32)
        edge = edge.astype(np.float32) # float64 causes error in cv2
        orimap = self.edge_detection.computeOrientation(edge)
        edge = self.edge_detection.edgesNms(edge, orimap)
        edge = (edge > 0.01).astype(np.float32)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image,
            'edge': edge
        }

class EndlessBIPEDv2Dataset(BIPEDv2Dataset):
    def __len__(self):
        return len(self.image_files) * 10000

    def get_by_idx(self, idx):
        idx = idx % len(self.image_files)
        return super().get_by_idx(idx)