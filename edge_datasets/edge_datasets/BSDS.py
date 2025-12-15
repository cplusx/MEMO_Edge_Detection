import os
import glob
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
import cv2
from .image_augmentor import ImageAugmentor

DATASET_DIR = 'edge_data/HED-BSDS'
DATASET_META = {
    'train': {
        'images_dir': '{}/train',
        'edge_dir': '{}/train'
    },
    'test': {
        'images_dir': '{}/test/',
        'edge_dir': '{}/gt/'
    }

}

class BSDSDataset(Dataset):
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
        if mode == 'train':
            self.image_files = sorted(glob.glob(os.path.join(images_dir, '**/*.jpg'), recursive=True))
            self.edge_files = sorted(glob.glob(os.path.join(edge_dir, '**/*.png'), recursive=True))
        else:
            self.image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
            self.edge_files = sorted(glob.glob(os.path.join(edge_dir, '*.mat')))

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
        if self.mode == 'train':
            edge = Image.open(edge_path).convert('L')
        else:
            mat_data = loadmat(edge_path)
            num_edges = len(mat_data['groundTruth'][0])
            edge_idx = np.random.randint(num_edges)
            edge = mat_data['groundTruth'][0][edge_idx][0][0][1]
            edge = Image.fromarray(edge * 255).convert('L')

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

