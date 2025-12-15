import os
import glob
import numpy as np
from PIL import Image, ImageOps
import scipy.sparse as sp
from .LAION_synthetic import LAIONSyntheticDataset, ImageAugmentor, EdgePostProcessing, colors_panel_36

class LAIONSyntheticQuantizedV2Dataset(LAIONSyntheticDataset):
    '''
    images should have: image_dir/batch*/images/xxxx.jpg
    edges should have: edge_dir/batch*/edge/quantize_{num_quantize_level}/edge (or edge_index, junctions)/xxxx.npz
    '''
    def __init__(
        self, 
        image_dir, 
        edge_dir,
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
        num_quantize_level=6, 
    ):
        self.image_dir = image_dir
        self.edge_dir = edge_dir
        self.image_and_edge_path = image_and_edge_path if image_and_edge_path is not None else os.path.join(edge_dir, f'image_edge_v2_path_quan{num_quantize_level}.txt')
        self.mode = mode
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.transpose_flip = transpose_flip
        self.vertical_flip = vertical_flip
        self.edge_quantize_level = num_quantize_level
        self.num_colors = num_quantize_level

        self.image_files = []
        self.edge_files = []
        if os.path.exists(self.image_and_edge_path):
            print(f'INFO: Load image and edge path from file {self.image_and_edge_path}')
            with open(self.image_and_edge_path, 'r') as f:
                for line in f:
                    image_path, edge_path = line.strip().split()
                    self.image_files.append(os.path.join(image_dir, image_path))
                    self.edge_files.append(os.path.join(edge_dir, edge_path))
        else:
            print(f'INFO: Generate image and edge path file {self.image_and_edge_path}')
            image_edge_file_path = self.generate_image_edge_file_path(image_dir, edge_dir, self.edge_quantize_level)
            with open(self.image_and_edge_path, 'w') as f:
                for image_path, edge_path in image_edge_file_path:
                    f.write(f"{image_path} {edge_path}\n")
                    self.image_files.append(os.path.join(image_dir, image_path))
                    self.edge_files.append(os.path.join(edge_dir, edge_path))

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

    def generate_image_edge_file_path(self, image_dir, edge_dir, quantize_level):
        all_images = glob.glob(os.path.join(image_dir, '**/*.jpg'), recursive=True)
        image_id_dict = {
            os.path.splitext(os.path.basename(image))[0]: image
            for image in all_images
        }
        all_edges = glob.glob(os.path.join(edge_dir, f'**/quantize_{quantize_level}/edge/*.npz'), recursive=True)
        all_edges = sorted(all_edges)
        image_edge_file_path = []

        for edge in all_edges:
            image_id = os.path.basename(edge).split('.')[0]
            if image_id not in image_id_dict:
                continue
            image_edge_file_path.append((
                image_id_dict[image_id].replace(image_dir, '').lstrip('/'),
                edge.replace(edge_dir, '').lstrip('/')
            ))

        return image_edge_file_path


    # dataset with connected components
    def get_by_idx(self, idx):
        # Load images
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]
        edge_label_path = edge_path.replace('/edge/', '/edge_index/')

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() # 0 - 36
        edge = np.where(edge > self.num_colors, self.num_colors, edge) # hack bug here for now
        edge = Image.fromarray(edge)

        edge_index = sp.load_npz(edge_label_path).toarray()

        num_components = edge_index.max()
        if num_components > 255:
            # pick components with largest area
            component_area = np.bincount(edge_index.flatten())
            component_area[0] = 0
            sorted_area = np.argsort(component_area)[::-1]
            mask = np.isin(edge_index, sorted_area[:255], invert=True)
            edge_index[mask] = 0
            # map index to 0-255
            unique_labels = np.unique(edge_index)
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            edge_index = np.vectorize(label_mapping.get)(edge_index).astype(np.uint8)

        edge_index = Image.fromarray(edge_index).convert('L')

        if self.augmentor is not None:
            # Pad images to square before augmentation
            _, edge_index = self.augmentor.pad_to_square(image, edge_index)
            image, edge = self.augmentor.pad_to_square(image, edge)

            # Compute and apply augmentation
            transform_params = self.augmentor.compute_transforms(image.size)
            image, edge, [edge_index] = self.augmentor.apply(image, edge, transform_params, scale_edge=False, others=[edge_index])

            if self.transpose_flip:
                if np.random.rand() > 0.5:
                    image = np.rot90(image, axes=(1, 0))
                    edge = np.rot90(edge, axes=(1, 0))
                    edge_index = np.rot90(edge_index, axes=(1, 0))
        else:
            # center crop
            image = ImageOps.fit(image, self.image_shape, Image.LANCZOS)
            edge = ImageOps.fit(edge, self.image_shape, Image.NEAREST)
            edge_index = ImageOps.fit(edge_index, self.image_shape, Image.NEAREST)

            image = np.array(image) / 255.0
            edge = np.array(edge)
            edge_index = np.array(edge_index)

        image = image.astype(np.float32)
        edge = edge.astype(np.uint8)
        edge_index = edge_index.astype(np.int32)

        # map index so it starts from 0 and increase by 1
        unique_labels = np.unique(edge_index)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        edge_index = np.vectorize(label_mapping.get)(edge_index).astype(np.uint8)

        return {
            'image_path': image_path,
            'edge_path': edge_path,
            'image': image, # h, w, 3
            'edge': edge, # h, w, 3
            'edge_index': edge_index, # h, w
        }

    # def __getitem__(self, idx):
    #     return self.get_by_idx(idx)

class LAIONSyntheticBinaryV2Dataset(LAIONSyntheticQuantizedV2Dataset):
    def get_by_idx(self, idx):
        res = super().get_by_idx(idx)
        res['edge'] = (res['edge'] > 0).astype(np.float32)  # convert to binary
        return res

class LAIONSyntheticColorizedV2Dataset(LAIONSyntheticQuantizedV2Dataset):

    def map_edge_to_color(self, edge):
        colorized_edge = colors_panel_36[edge]
        return colorized_edge

    def get_by_idx(self, idx):
        image_path = self.image_files[idx]
        edge_path = self.edge_files[idx]

        image = Image.open(image_path).convert('RGB')
        edge = sp.load_npz(edge_path).toarray() 
        edge = self.map_edge_to_color(edge)
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
            'edge': edge, # h, w, 3
        }