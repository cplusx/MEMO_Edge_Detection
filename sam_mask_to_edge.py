import os
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, binary_erosion, disk
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from itertools import product
from tqdm import tqdm
import scipy.sparse as sp
from glob import glob
import json
from edge_eval_python.impl.correspond_pixels import correspond_pixels
from skimage.morphology import thin
from edge_datasets.edge_datasets.image_augmentor import EdgePostProcessing
from misc_utils.edge_refiner import split_connected_components, decide_value_and_confidence, extract_and_count_values


def map_edge_to_bucket(edge, num_buckets):
    buckets = np.zeros_like(edge)
    num_edge_each_bucket = 18 // num_buckets
    buckets[edge > 0] = ((edge[edge > 0] - 1) // num_edge_each_bucket) + 1
    return buckets


def process_one_bucket_setting(edge):
    labeled_edges, junctions, endpoints = split_connected_components(edge)
    label_canvas = np.zeros_like(labeled_edges)
    conf_canvas = np.zeros_like(labeled_edges)
    num_labels = labeled_edges.max()

    for label in range(1, num_labels + 1):
        mask = labeled_edges == label
        if mask.sum() <= 1:
            continue
        _, count_values = extract_and_count_values(edge, mask)
        value, conf = decide_value_and_confidence(count_values)

        label_canvas[mask] = value
        conf_canvas[mask] = conf
    return labeled_edges, label_canvas, conf_canvas, junctions, endpoints

def inference_config(image, model, points_per_side_list, stability_score_thresh_list, crop_n_layers_list):
    edges = []
    for i, (points_per_side, stability_score_thresh, crop_n_layers) in enumerate(tqdm(product(points_per_side_list, stability_score_thresh_list, crop_n_layers_list))):
        try:
            mask_generator = SAM2AutomaticMaskGenerator(
                model,
                points_per_side=points_per_side,
                points_per_batch=128,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=2,
            )
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
                masks = mask_generator.generate(image)
            masks_polished = polish_masks(masks)
            edge = masks_to_edge_map(masks_polished, edge_width=1)

            edges.append(edge)
        except Exception as e:
            print(f'Error in inference config {i}: {e}')
            continue

    return edges


def polish_masks(masks, gap_threshold=2):
    """
    Assign unassigned pixels in narrow gaps between masks to the nearest mask.

    Parameters:
    - masks: list of dictionaries, each containing a 'segmentation' key with a boolean array.
    - gap_threshold: Maximum width of gaps to be filled between masks.

    Returns:
    - polished_masks: list of dictionaries with updated 'segmentation' arrays.
    """
    if not masks:
        raise ValueError("The masks list is empty.")
    
    # Sort masks by area in descending order
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Determine the shape of the segmentation maps
    mask_shape = masks[0]['segmentation'].shape

    # Initialize label image
    label_image = np.zeros(mask_shape, dtype=np.int32)

    # Assign unique labels to each mask
    for i, mask in enumerate(masks):
        label = i + 1  # Labels start from 1
        segmentation = mask['segmentation']
        label_image[segmentation] = label

    # Find unassigned pixels
    unassigned = label_image == 0

    # If there are no unassigned pixels, return the original masks
    if not np.any(unassigned):
        return masks

    # Create a binary mask of all assigned regions
    assigned_regions = label_image > 0

    # Dilate the assigned regions to find narrow gaps
    dilated_regions = binary_dilation(assigned_regions, disk(gap_threshold))
    narrow_gaps = dilated_regions & unassigned

    # Compute distance transform on narrow gaps
    # Get indices of nearest assigned pixels
    distance, indices = distance_transform_edt(~assigned_regions, return_indices=True)

    # For narrow gap pixels, get the labels of nearest assigned pixel
    nearest_i = indices[0][narrow_gaps]
    nearest_j = indices[1][narrow_gaps]
    nearest_labels = label_image[nearest_i, nearest_j]

    # Assign the narrow gap pixels to the nearest labels
    label_image[narrow_gaps] = nearest_labels

    # Update the masks with the new segmentations
    num_labels = len(masks)
    polished_masks = []
    for i, mask in enumerate(masks):
        label = i + 1
        new_segmentation = label_image == label

        # Update the mask's segmentation
        new_mask = mask.copy()
        new_mask['segmentation'] = new_segmentation

        # Optionally, update 'area' and 'bbox' if necessary
        new_mask['area'] = np.sum(new_segmentation)
        rows = np.any(new_segmentation, axis=1)
        cols = np.any(new_segmentation, axis=0)
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            new_mask['bbox'] = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
        else:
            new_mask['bbox'] = [0.0, 0.0, 0.0, 0.0]

        polished_masks.append(new_mask)

    return polished_masks

def masks_to_edge_map(masks, edge_width=1):
    """
    Converts a list of masks to an edge map with configurable edge width.

    Parameters:
    - masks: list of dictionaries, each containing a 'segmentation' key with a boolean array.
    - edge_width: integer specifying the width of the edges (default is 1).

    Returns:
    - edge_map: numpy array with edges marked as True (edges) and False (non-edges).
    """
    if not masks:
        raise ValueError("The masks list is empty.")
    # masks = sorted(masks, key=lambda x: x['area'], reverse=True)

    # Determine the shape of the segmentation maps
    mask_shape = masks[0]['segmentation'].shape

    # Initialize an empty edge map
    edge_map = np.zeros(mask_shape, dtype=bool)

    # Structuring element for dilation to adjust edge width
    selem = disk(max(edge_width // 2, 1))

    for mask in masks:
        segmentation = mask['segmentation']

        # Check if the segmentation map is boolean
        if segmentation.dtype != bool:
            segmentation = segmentation.astype(bool)

        # Find edges by subtracting the eroded mask from the original mask
        eroded_seg = binary_erosion(segmentation)
        edges = segmentation & ~eroded_seg

        # Adjust edge width if necessary
        if edge_width > 1:
            edges = binary_dilation(edges, footprint=selem)

        # Combine edges into the edge map
        edge_map |= edges

    return edge_map

def get_image_paths_by_dataset(dataset_name):
    # dataset name or image dir
    image_paths = sorted(glob(f'{dataset_name}/*.jpg', recursive=True)) + sorted(glob(f'{dataset_name}/*.png', recursive=True))
    image_names = [os.path.basename(image_path).split('.')[0] for image_path in image_paths]
    return list(zip(image_paths, image_names))

def save_processed_edges(edge_index, edge, junctions, endpoints, save_dir, image_name):
    '''
    edge_index: the index of edge components
    edge: the quantized edge map
    '''
    edge_save_path = os.path.join(save_dir, 'edge', image_name)
    edge_index_save_path = os.path.join(save_dir, 'edge_index', image_name)
    junctions_save_path = os.path.join(save_dir, 'junctions', image_name + '.json')
    os.makedirs(os.path.dirname(edge_save_path), exist_ok=True)
    sp.save_npz(edge_save_path, sp.csr_matrix(edge, dtype=np.uint8))
    os.makedirs(os.path.dirname(edge_index_save_path), exist_ok=True)
    sp.save_npz(edge_index_save_path, sp.csr_matrix(edge_index, dtype=np.int16))
    os.makedirs(os.path.dirname(junctions_save_path), exist_ok=True)
    with open(junctions_save_path, 'w') as f:
        json.dump({
            'junctions': junctions,
            'endpoints': endpoints
        }, f, indent=4)

def argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='', help='dataset name or image dir')
    parser.add_argument('--save_dir', type=str, default='', help='save dir')
    parser.add_argument('--start_idx', type=int, default=0, help='start index')
    parser.add_argument('--num_images', type=int, default=100, help='number of images to process')
    parser.add_argument('--max_size', type=int, default=3000, help='max size of the image')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    '''
    python sam_mask_to_edge.py --dataset_name /home/devdata/laion/edge_detection/batch1/images --save_dir laion_edge_v2/batch1 --start_idx 0 --num_images 10
    '''
    args = argparse()
    dataset_name = args.dataset_name
    save_dir = args.save_dir
    start_idx = args.start_idx
    num_images = args.num_images
    quantize_list = [6]
    edge_post_processing = EdgePostProcessing(min_connected_component=8)

    os.makedirs(save_dir, exist_ok=True)

    image_paths = get_image_paths_by_dataset(dataset_name)[start_idx:start_idx+num_images]


    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-large",
        compile_image_encoder=True
    )

    points_per_side_list = [64] # larger number, more edges
    stability_score_thresh_list = [0.85] # smaller then number, more edges
    crop_n_layers_list = [1]

    for image_path, image_name in tqdm(image_paths):

        if os.path.exists(f'{save_dir}/quantize_6/edge/{image_name}.npz'):
            print('Already exists:', image_name)
            continue
        try:
            image = Image.open(image_path).convert('RGB')
            # resize the image if it is too large
            if image.size[0] > args.max_size or image.size[1] > args.max_size:
                image.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
            image = np.array(image)
            edges = inference_config(image, predictor.model, points_per_side_list, stability_score_thresh_list, crop_n_layers_list)
            edges = np.array(edges)

            canvas = np.zeros_like(edges[0], dtype=np.float32)
            for edge in edges:
                this_match, _, _, _ = correspond_pixels(thin(edges[0]), thin(edge), 0.0075)
                canvas += (this_match > 0).astype(np.float32)

            for num_buckets in quantize_list:
                this_save_dir = os.path.join(save_dir, f'quantize_{num_buckets}')
                edge_quantized = map_edge_to_bucket(canvas, num_buckets)
                edge_quantized = edge_post_processing(edge_quantized)
                indexed_edges, label, _, junctions, endpoints = process_one_bucket_setting(edge_quantized)
                save_processed_edges(indexed_edges, label, junctions, endpoints, this_save_dir, image_name)
        except Exception as e:
            print(f'Error: {image_name}, {e}')
            continue
