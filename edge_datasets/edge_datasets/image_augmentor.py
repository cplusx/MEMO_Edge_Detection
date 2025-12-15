import random
import torchvision.transforms.functional as F
from scipy.ndimage import distance_transform_edt
import math
import numpy as np
from PIL import Image
import cv2

class ImageAugmentor:
    def __init__(
        self, 
        image_shape=(256, 256), 
        size_range=(0.8, 1.0),
        aspect_ratio_range=(0.9, 1.1), 
        perspective_range=0.0, 
        rotation_range=360,
        vertical_flip=True,
        horizontal_flip=True
    ):
        self.image_shape = image_shape  # (height, width)
        self.size_range = size_range
        self.aspect_ratio_range = aspect_ratio_range
        self.perspective_range = perspective_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def pad_to_square(self, image, edge):
        """Pad the image and edge to make them square."""
        max_dim = max(image.size)
        padding = (
            (max_dim - image.width) // 2, 
            (max_dim - image.height) // 2, 
            (max_dim - image.width + 1) // 2, 
            (max_dim - image.height + 1) // 2
        )
        # padding with opencv
        image = np.pad(np.array(image), ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)), mode='symmetric')
        if len(np.array(edge).shape) == 3:
            edge = np.pad(np.array(edge), ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)), mode='symmetric')
        else:
            edge = np.pad(np.array(edge), ((padding[1], padding[3]), (padding[0], padding[2])), mode='symmetric')
        image = Image.fromarray(image)
        edge = Image.fromarray(edge)
        return image, edge

    def compute_transforms(self, original_size):
        transforms = {}
        # Random scale
        scale_factor = random.uniform(*self.size_range)
        new_w = int(original_size[0] * scale_factor)
        new_h = int(original_size[1] * scale_factor)
        transforms['scale'] = (new_w, new_h)

        # Random aspect ratio
        aspect_ratio = random.uniform(*self.aspect_ratio_range)
        new_w_ar = int(new_w * aspect_ratio)
        transforms['aspect_ratio'] = (new_w_ar, new_h)

        # Rotation degree
        max_rotation = self.get_max_rotation((new_w_ar, new_h), self.image_shape)
        rotation_degree = random.uniform(-min(self.rotation_range, max_rotation),
                                         min(self.rotation_range, max_rotation))
        transforms['rotation'] = rotation_degree

        # Perspective transformation
        if self.perspective_range > 0:
            startpoints, endpoints = self.get_perspective_params((new_w_ar, new_h), self.perspective_range)
            coeffs = self.get_perspective_coeffs(startpoints, endpoints)
            transforms['perspective'] = {'coeffs': coeffs, 'startpoints': startpoints, 'endpoints': endpoints}
        else:
            transforms['perspective'] = None

        # Random flips
        transforms['horizontal_flip'] = self.horizontal_flip and random.random() > 0.5
        transforms['vertical_flip'] = self.vertical_flip and random.random() > 0.5

        return transforms

    def apply(self, image, edge, transform_params, scale_edge=True, others=[]):
        # Create a validation mask for the image
        valid_mask = Image.fromarray((np.array(image.convert('L')) > 0).astype(np.uint8) * 255)
        # set pixels near the border to 0 (if distance to border < 0.5 * min(image_shape))
        valid_mask = np.array(valid_mask)
        valid_mask[0:int(self.image_shape[0] * 0.125), :] = 0
        valid_mask[-int(self.image_shape[0] * 0.125):, :] = 0
        valid_mask[:, 0:int(self.image_shape[1] * 0.125)] = 0
        valid_mask[:, -int(self.image_shape[1] * 0.125):] = 0
        valid_mask = Image.fromarray(valid_mask)

        # Apply scaling
        new_w, new_h = transform_params['scale']
        image = image.resize((new_w, new_h), resample=Image.BILINEAR)
        edge = edge.resize((new_w, new_h), resample=Image.BILINEAR)
        valid_mask = valid_mask.resize((new_w, new_h), resample=Image.NEAREST)
        if len(others) > 0:
            others = [other.resize((new_w, new_h), resample=Image.BILINEAR) for other in others]

        # Apply aspect ratio adjustment
        new_w_ar, new_h = transform_params['aspect_ratio']
        image = image.resize((new_w_ar, new_h), resample=Image.BILINEAR)
        edge = edge.resize((new_w_ar, new_h), resample=Image.BILINEAR)
        valid_mask = valid_mask.resize((new_w_ar, new_h), resample=Image.NEAREST)
        if len(others) > 0:
            others = [other.resize((new_w_ar, new_h), resample=Image.BILINEAR) for other in others]

        # Apply rotation
        rotation_degree = transform_params['rotation']
        image = image.rotate(rotation_degree, resample=Image.BILINEAR, expand=True)
        edge = edge.rotate(rotation_degree, resample=Image.BILINEAR, expand=True)
        valid_mask = valid_mask.rotate(rotation_degree, resample=Image.NEAREST, expand=True)
        if len(others) > 0:
            others = [other.rotate(rotation_degree, resample=Image.BILINEAR, expand=True) for other in others]

        # Apply perspective transform if specified
        if transform_params['perspective'] is not None:
            perspective_params = transform_params['perspective']
            coeffs = perspective_params['coeffs']
            transform_matrix = Image.PERSPECTIVE
            # Apply the same transform to all images
            image = image.transform(image.size, transform_matrix, coeffs, resample=Image.BILINEAR)
            edge = edge.transform(edge.size, transform_matrix, coeffs, resample=Image.BILINEAR)
            valid_mask = valid_mask.transform(valid_mask.size, transform_matrix, coeffs, resample=Image.NEAREST)
            if len(others) > 0:
                others = [other.transform(other.size, transform_matrix, coeffs, resample=Image.BILINEAR) for other in others]

        # Apply random horizontal flip
        if transform_params['horizontal_flip']:
            image = F.hflip(image)
            edge = F.hflip(edge)
            valid_mask = F.hflip(valid_mask)
            if len(others) > 0:
                others = [F.hflip(other) for other in others]

        # Apply random vertical flip
        if transform_params['vertical_flip']:
            image = F.vflip(image)
            edge = F.vflip(edge)
            valid_mask = F.vflip(valid_mask)
            if len(others) > 0:
                others = [F.vflip(other) for other in others]

        # Random crop to desired image_shape ensuring valid region
        if len(others) > 0:
            image, edge, valid_mask, others = self.random_crop_valid_region(image, edge, valid_mask, self.image_shape, others)
        else:
            image, edge, valid_mask = self.random_crop_valid_region(image, edge, valid_mask, self.image_shape)

        # Convert to tensors
        image = np.array(image) / 255.0
        if scale_edge:
            edge = np.array(edge) / 255.0
        else:
            edge = np.array(edge)
        if len(others) > 0:
            others = [np.array(other) for other in others]

        if len(others) > 0:
            return image, edge, others
        else:
            return image, edge

    def random_crop_valid_region(self, image, edge, valid_mask, crop_size, others=[]):
        """Randomly crop the image and edge to the given size, ensuring valid regions."""
        crop_w, crop_h = crop_size[1], crop_size[0]  # (width, height)
        img_w, img_h = image.size

        # Compute valid_mask_np
        valid_mask_np = np.array(valid_mask) > 0

        # Compute the distance transform
        distance = distance_transform_edt(valid_mask_np)

        # Compute minimum distance threshold
        min_distance = (np.sqrt(2) / 2) * min(crop_h, crop_w)

        # Find coordinates of pixels whose distance to border is greater than min_distance
        valid_coords = np.argwhere(distance > min_distance)

        if valid_coords.size == 0:
            # If no valid region found, return a centered crop
            # print("File: ", __file__, "No valid region found. Returning centered crop.")
            left = max((img_w - crop_w) // 2, 0)
            upper = max((img_h - crop_h) // 2, 0)
        else:
            # Get y and x coordinates of valid pixels
            y_coords, x_coords = valid_coords[:, 0], valid_coords[:, 1]
            coord_idx = random.randint(0, len(y_coords) - 1)
            center_x, center_y = x_coords[coord_idx], y_coords[coord_idx]
            left = max(center_x - crop_w // 2, 0)
            upper = max(center_y - crop_h // 2, 0)

        # Crop the images
        if left + crop_w > img_w or upper + crop_h > img_h:
            image = image.crop((left, upper, img_w, img_h))
            edge = edge.crop((left, upper, img_w, img_h))
            valid_mask = valid_mask.crop((left, upper, img_w, img_h))
            image = image.resize((crop_w, crop_h), resample=Image.BILINEAR)
            edge = edge.resize((crop_w, crop_h), resample=Image.BILINEAR)
            valid_mask = valid_mask.resize((crop_w, crop_h), resample=Image.NEAREST)
            if len(others) > 0:
                others = [other.crop((left, upper, img_w, img_h)).resize((crop_w, crop_h), resample=Image.BILINEAR) for other in others]
        else:
            image = image.crop((left, upper, left + crop_w, upper + crop_h))
            edge = edge.crop((left, upper, left + crop_w, upper + crop_h))
            valid_mask = valid_mask.crop((left, upper, left + crop_w, upper + crop_h))
            if len(others) > 0:
                others = [other.crop((left, upper, left + crop_w, upper + crop_h)) for other in others]

        if len(others) > 0:
            return image, edge, valid_mask, others
        return image, edge, valid_mask

    def get_max_rotation(self, img_size, target_size):
        # This function calculates the maximum rotation angle that ensures the target_size fits within the image
        w, h = img_size
        tw, th = target_size

        max_angle_w = math.degrees(math.atan2(h, w - tw)) if w > tw else 90
        max_angle_h = math.degrees(math.atan2(w, h - th)) if h > th else 90

        max_angle = min(max_angle_w, max_angle_h)
        return max_angle

    def get_perspective_params(self, size, perspective_range):
        w, h = size
        max_disp_x = w * perspective_range
        max_disp_y = h * perspective_range

        # Define the four corners
        tl = (random.uniform(0, max_disp_x), random.uniform(0, max_disp_y))
        tr = (w - random.uniform(0, max_disp_x), random.uniform(0, max_disp_y))
        br = (w - random.uniform(0, max_disp_x), h - random.uniform(0, max_disp_y))
        bl = (random.uniform(0, max_disp_x), h - random.uniform(0, max_disp_y))

        startpoints = [(0, 0), (w, 0), (w, h), (0, h)]
        endpoints = [tl, tr, br, bl]

        return startpoints, endpoints

    def get_perspective_coeffs(self, startpoints, endpoints):
        matrix = []
        for p1, p2 in zip(startpoints, endpoints):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0,
                           -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1,
                           -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.array(matrix, dtype=np.float32)
        B = np.array(endpoints).reshape(8)

        res = np.linalg.lstsq(A, B, rcond=None)[0]
        return res

class ImageCropper:
    def __init__(
        self, 
        image_shape=(256, 256),
        vertical_flip=True,
        horizontal_flip=True
    ):
        self.image_shape = image_shape
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

    def pad_to_square(self, image, edge):
        """Pad the image and edge to make them square."""
        max_dim = max(image.size)
        padding = (
            (max_dim - image.width) // 2, 
            (max_dim - image.height) // 2, 
            (max_dim - image.width + 1) // 2, 
            (max_dim - image.height + 1) // 2
        )
        # padding with opencv
        image = np.pad(np.array(image), ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)), mode='symmetric')
        edge = np.pad(np.array(edge), ((padding[1], padding[3]), (padding[0], padding[2])), mode='symmetric')
        image = Image.fromarray(image)
        edge = Image.fromarray(edge)
        return image, edge

    def __call__(self, image, edge):
        image = np.array(image) / 255.0
        edge = np.array(edge) / 255.0

        h, w = image.shape[:2]
        h_start = random.randint(0, h - self.image_shape[0])
        w_start = random.randint(0, w - self.image_shape[1])
        h_end = h_start + self.image_shape[0]
        w_end = w_start + self.image_shape[1]

        image = image[h_start:h_end, w_start:w_end]
        edge = edge[h_start:h_end, w_start:w_end]

        if self.horizontal_flip and random.random() > 0.5:
            image = np.fliplr(image)
            edge = np.fliplr(edge)

        if self.vertical_flip and random.random() > 0.5:
            image = np.flipud(image)
            edge = np.flipud(edge)

        return image, edge

OPENCV_EDGE_MODEL = 'opencv_edge/model.yml.gz'

class EdgePostProcessing():
    def __init__(self, min_connected_component=8, edge_detector_path=OPENCV_EDGE_MODEL):
        self.edge_detector_path = edge_detector_path
        self.edge_detection = cv2.ximgproc.createStructuredEdgeDetection(edge_detector_path)
        self.min_connected_component = min_connected_component

    def edge_thinning(self, edge):
        if edge.max() <= 1e-4: # if edge is all zeros,
            return edge
        edge_float = (edge / edge.max()).astype(np.float32)
        orimap = self.edge_detection.computeOrientation(edge_float)
        edge_float = self.edge_detection.edgesNms(edge_float, orimap)
        edge_to_keep_mask = (edge_float > 0.01).astype(np.float32)
        edge_thin = np.where(edge_to_keep_mask > 0, edge, 0)
        return edge_thin

    def remove_small_area(self, edge):
        edge_binary = (edge > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_binary, connectivity=8)
        small_area_labels = np.where(stats[:, cv2.CC_STAT_AREA] < self.min_connected_component)[0]
        # Create a mask of all pixels belonging to the small area labels
        mask = np.isin(labels, small_area_labels)

        # Set all these pixels in the edge array to 0
        edge = np.where(mask, 0, edge)
        return edge

    def __call__(self, edge):
        edge_thin = self.edge_thinning(edge)
        edge_thin = self.remove_small_area(edge_thin)
        return edge_thin

