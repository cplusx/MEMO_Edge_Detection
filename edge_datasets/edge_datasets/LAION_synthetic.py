import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from .image_augmentor import ImageAugmentor, EdgePostProcessing, ImageCropper
import scipy.sparse as sp
import cv2
import json

DATASET_DIR = '/home/devdata/laion/for_edge_detection'

class LAIONSyntheticDataset(Dataset):
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
        horizontal_flip=True,
        transpose_flip=True
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.transpose_flip = transpose_flip
        self.vertical_flip = vertical_flip

        # Build the image and edge map paths
        images_dir = os.path.join(root_dir, 'image')
        edge_dir = os.path.join(root_dir, 'edges')

        # Get list of image files
        self.edge_files = sorted(glob.glob(os.path.join(edge_dir, '*.npz')))
        self.image_files = [
            os.path.join(
                images_dir, 
                os.path.basename(edge_file).replace('.npz', '.jpg')
            ) for edge_file in self.edge_files
        ]

        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]
            self.edge_files = self.edge_files[:max_samples]

        assert len(self.image_files) == len(self.edge_files), "Number of images and edge maps must be equal."

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

        self.edge_post_processing = EdgePostProcessing(
            min_connected_component=8
        )

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
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = self.edge_post_processing(edge)
        edge = 255. * edge / 36.0 # scale to 0-255
        edge = Image.fromarray(edge)

        if self.augmentor is not None:
            # Pad images to square before augmentation
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge = self.augmentor.apply(image, edge, transform_params)

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.BILINEAR)

            image = np.array(image) / 255.0
            edge = np.array(edge) / 255.0

        image = image.astype(np.float32)
        edge = edge.astype(np.float32) # float64 causes error in cv2

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge # h, w
        }

class LAIONSyntheticBucketMapDataset(LAIONSyntheticDataset):
    def __init__(self, *args, num_buckets=6, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_buckets = num_buckets

    def map_edge_to_bucket(self, edge):
        buckets = np.zeros_like(edge)
        num_edge_each_bucket = 36 // self.num_buckets
        buckets[edge > 0] = ((edge[edge > 0] - 1) // num_edge_each_bucket) + 1
        return buckets

    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = self.map_edge_to_bucket(edge)
        edge = self.edge_post_processing(edge)
        edge = 255. * edge / self.num_buckets # scale to 0-255
        edge = Image.fromarray(edge)

        if self.augmentor is not None:
            # Pad images to square before augmentation
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge = self.augmentor.apply(image, edge, transform_params)

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.BILINEAR)

            image = np.array(image) / 255.0
            edge = np.array(edge) / 255.0

        image = image.astype(np.float32)
        edge = edge.astype(np.float32) # float64 causes error in cv2

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge # h, w
        }

colors_panel_36 = np.array([
    (0, 0, 0),        # 0.  Black
    (255, 0, 0),      # 1.  Bright Red
    (0, 255, 0),      # 2.  Bright Green
    (0, 0, 255),      # 3.  Bright Blue
    (255, 255, 0),    # 4.  Yellow
    (255, 0, 255),    # 5.  Magenta
    (0, 255, 255),    # 6.  Cyan
    (128, 128, 128),  # 7.  Gray
    (128, 0, 0),      # 8.  Maroon
    (0, 128, 0),      # 9.  Dark Green
    (0, 0, 128),      # 10. Navy
    (128, 128, 0),    # 11. Olive
    (128, 0, 128),    # 12. Purple
    (0, 128, 128),    # 13. Teal
    (255, 128, 0),    # 14. Orange
    (255, 0, 128),    # 15. Deep Pink
    (128, 255, 0),    # 16. Yellow-Green
    (0, 255, 128),    # 17. Spring Green
    (128, 0, 255),    # 18. Violet
    (0, 128, 255),    # 19. Azure
    (255, 128, 128),  # 20. Light Coral
    (128, 255, 128),  # 21. Light Green
    (128, 128, 255),  # 22. Light Slate
    (255, 255, 128),  # 23. Light Yellow
    (255, 128, 255),  # 24. Light Magenta
    (128, 255, 255),  # 25. Light Cyan
    (192, 64, 0),     # 26. Brownish Orange
    (192, 0, 64),     # 27. Crimson
    (64, 192, 0),     # 28. Lime
    (0, 192, 64),     # 29. Emerald
    (64, 0, 192),     # 30. Indigo
    (0, 64, 192),     # 31. Cerulean
    (192, 192, 64),   # 32. Khaki
    (192, 64, 192),   # 33. Orchid
    (64, 192, 192),   # 34. Turquoise
    (192, 192, 192),  # 35. Silver
    (64, 64, 64)      # 36. Dark Gray
]).astype(np.uint8)

class LAIONSyntheticPseudoColorDataset(LAIONSyntheticDataset):
    def __init__(
        self, 
        root_dir, 
        image_and_edge_path=None,
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
        num_colors=6, 
    ):
        self.root_dir = root_dir
        self.image_and_edge_path = image_and_edge_path if image_and_edge_path is not None else os.path.join(root_dir, f'image_edge_path_quan{num_colors}.txt')
        self.mode = mode
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.transpose_flip = transpose_flip
        self.vertical_flip = vertical_flip
        self.edge_quantize_level = num_colors
        self.num_colors = num_colors

        self.image_files = []
        self.edge_files = []
        if os.path.exists(self.image_and_edge_path):
            print(f'INFO: Load image and edge path from file {self.image_and_edge_path}')
            with open(self.image_and_edge_path, 'r') as f:
                for line in f:
                    image_path, edge_path = line.strip().split()
                    self.image_files.append(os.path.join(root_dir, image_path))
                    self.edge_files.append(os.path.join(root_dir, edge_path))
        else:
            print(f'INFO: Generate image and edge path file {self.image_and_edge_path}')
            image_edge_file_path = self.generate_image_edge_file_path(root_dir, self.edge_quantize_level)
            with open(self.image_and_edge_path, 'w') as f:
                for image_path, edge_path in image_edge_file_path:
                    f.write(f"{image_path} {edge_path}\n")
                    self.image_files.append(os.path.join(root_dir, image_path))
                    self.edge_files.append(os.path.join(root_dir, edge_path))

        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]
            self.edge_files = self.edge_files[:max_samples]

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

        self.edge_post_processing = EdgePostProcessing(
            min_connected_component=8
        )

    def generate_image_edge_file_path(self, root_dir, quantize_level):
        all_images = glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True)
        image_id_dict = {
            os.path.splitext(os.path.basename(image))[0]: image
            for image in all_images
        }
        all_edges = glob.glob(os.path.join(root_dir, f'**/quantize_{quantize_level}/edge/*.npz'), recursive=True)
        all_edges = sorted(all_edges)
        image_edge_file_path = []

        for edge in all_edges:
            image_id = os.path.basename(edge).split('.')[0]
            if image_id not in image_id_dict:
                continue
            image_edge_file_path.append((
                image_id_dict[image_id].replace(root_dir, '').lstrip('/'),
                edge.replace(root_dir, '').lstrip('/')
            ))

        return image_edge_file_path

    def map_edge_to_color(self, edge):
        colorized_edge = colors_panel_36[edge]
        return colorized_edge

    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() 
        edge = self.map_edge_to_color(edge)
        edge = Image.fromarray(edge)

        if self.augmentor is not None:
            # Pad images to square before augmentation
            # _, conf = self.augmentor.pad_to_square(image, conf)
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            # _, conf = self.augmentor.apply(image, conf, transform_params)
            image, edge = self.augmentor.apply(image, edge, transform_params)

            # conf = (conf * 255.).astype(np.uint8) # the apply() will modify image to /255

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
                    # conf = np.rot90(edge, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.BILINEAR)
            # conf = ImageOps.fit(edge, self.image_shape, Image.BILINEAR)

            image = np.array(image) / 255.0
            edge = np.array(edge) / 255.0
            # conf = np.array(conf)

        image = image.astype(np.float32)
        edge = edge.astype(np.float32) # float64 causes error in cv2

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge, # h, w, 3
            # 'conf': conf, # h, w
        }

    # def __getitem__(self, idx):
    #     return self.get_by_idx(idx)

class LAIONSyntheticQuantizedDataset(LAIONSyntheticPseudoColorDataset):
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = np.where(edge > self.num_colors, self.num_colors, edge) # hack bug here for now
        edge = Image.fromarray(edge)

        if self.augmentor is not None:
            # Pad images to square before augmentation
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge = self.augmentor.apply(image, edge, transform_params, scale_edge=False)

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.NEAREST)

            image = np.array(image) / 255.0
            edge = np.array(edge)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge, # h, w, 3
        }

    # def __getitem__(self, idx):
    #     return self.get_by_idx(idx)

class LAIONSyntheticQuantizedLabeledDataset(LAIONSyntheticQuantizedDataset):
    # dataset with connected components
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]
        edge_label_path = edge_path.replace('/edge/', '/edge_labels/')

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = np.where(edge > self.num_colors, self.num_colors, edge) # hack bug here for now
        edge = Image.fromarray(edge)

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

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge, [edge_labels] = self.augmentor.apply(image, edge, transform_params, scale_edge=False, others=[edge_labels])

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
                    edge_labels = np.rot90(edge_labels, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.NEAREST)
            edge_labels = ImageOps.fit(edge_labels, self.image_shape, Image.NEAREST)

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge_labels = np.array(edge_labels)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)
        edge_labels = edge_labels.astype(np.int32)

        # map index so it starts from 0 and increase by 1
        unique_labels = np.unique(edge_labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        edge_labels = np.vectorize(label_mapping.get)(edge_labels).astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge, # h, w, 3
            'edge_labels': edge_labels, # h, w
        }

    # def __getitem__(self, idx):
    #     return self.get_by_idx(idx)
    
def get_uncertainty_map_by_junction(height, width, junctions, nearby_pixels=3):
    junction_canvas = np.zeros((height, width), dtype=np.float32)
    junction_canvas[junctions[:, 0], junctions[:, 1]] = 1
    kernel_size = 2 * nearby_pixels + 1
    kernel = np.ones((kernel_size, kernel_size), np.float32)
    junction_canvas_filtered = cv2.filter2D(junction_canvas, -1, kernel)

    return (junction_canvas_filtered > 3).astype(np.float32)

class LAIONSyntheticQuantizedLabeledJunctionUncertaintyDataset(LAIONSyntheticQuantizedDataset):
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]
        edge_label_path = edge_path.replace('/edge/', '/edge_labels/')
        junction_file_path = edge_label_path.replace('/edge_labels/', '/junctions/').replace('.npz', '.json')

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = np.where(edge > self.num_colors, self.num_colors, edge) # hack bug here for now
        edge = Image.fromarray(edge)
        edge_labels = sp.load_npz(edge_label_path).toarray()
        with open(junction_file_path, 'r') as f:
            junctions = np.array(json.load(f)['junctions'])

        junction_uncertainty = 1 - get_uncertainty_map_by_junction(image.size[1], image.size[0], junctions) # 1 for certain, 0 for uncertain
        junction_uncertainty = Image.fromarray((junction_uncertainty * 255).astype(np.uint8))

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
            _, junction_uncertainty = self.augmentor.pad_to_square(image, junction_uncertainty)
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            # _, edge_labels = self.augmentor.apply(image, edge_labels, transform_params, scale_edge=False)
            image, edge, [edge_labels, junction_uncertainty] = self.augmentor.apply(image, edge, transform_params, scale_edge=False, others=[edge_labels, junction_uncertainty])
            junction_uncertainty = junction_uncertainty / 255.0

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
                    edge_labels = np.rot90(edge_labels, axes=(1, 0))
                    junction_uncertainty = np.rot90(junction_uncertainty, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.NEAREST)
            edge_labels = ImageOps.fit(edge_labels, self.image_shape, Image.NEAREST)
            junction_uncertainty = ImageOps.fit(junction_uncertainty, self.image_shape, Image.LANCZOS)

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge_labels = np.array(edge_labels)
            junction_uncertainty = np.array(junction_uncertainty) / 255.0

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)
        edge_labels = edge_labels.astype(np.int32)
        junction_uncertainty = junction_uncertainty.astype(np.float32)

        # map index so it starts from 0 and increase by 1
        unique_labels = np.unique(edge_labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        edge_labels = np.vectorize(label_mapping.get)(edge_labels).astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge, # h, w, 3
            'edge_labels': edge_labels, # h, w
            'junction_uncertainty': junction_uncertainty, # h, w
        }

    # def __getitem__(self, idx):
    #     return self.get_by_idx(idx)