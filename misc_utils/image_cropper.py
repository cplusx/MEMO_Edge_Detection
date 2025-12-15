import torch
import numpy as np
import torch.nn.functional as F

class ImageCropper:
    def __init__(self):
        pass

    def get_cropping_configs(self, num_augs, image_shape, aspect_ratio_range, min_size, max_size):
        """
        Generate cropping configurations.

        Args:
            num_augs (int): Number of cropping configs to generate.
            image_shape (tuple): (height, width) of the image.
            aspect_ratio_range (tuple): (min_ar, max_ar), aspect ratio range.
            min_size (float): Min size fraction (between 0 and 1).
            max_size (float): Max size fraction (between 0 and 1).

        Returns:
            cropping_configs (list): List of cropping configs, each is (x1, y1, crop_width, crop_height)
        """
        H, W = image_shape
        min_ar, max_ar = aspect_ratio_range
        cropping_configs = []
        max_attempts = num_augs * 10  # To prevent infinite loops
        attempts = 0

        while len(cropping_configs) < num_augs and attempts < max_attempts:
            attempts += 1
            area_fraction = np.random.uniform(min_size, max_size)
            target_area = area_fraction * H * W
            aspect_ratio = np.random.uniform(min_ar, max_ar)

            crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(np.sqrt(target_area / aspect_ratio)))

            if crop_width <= W and crop_height <= H:
                x1 = np.random.randint(0, W - crop_width + 1)
                y1 = np.random.randint(0, H - crop_height + 1)
                cropping_configs.append((x1, y1, crop_width, crop_height))
            else:
                continue  # Skip invalid configs

        if len(cropping_configs) < num_augs:
            print("Warning: Could not generate the desired number of cropping configurations.")
        return cropping_configs

    def apply_cropping_configs(self, image, cropping_configs):
        """
        Apply cropping configurations to an image.

        Args:
            image (torch.Tensor): Image tensor of shape (1, 3, H, W) or (3, H, W).
            cropping_configs (list): List of cropping configurations.

        Returns:
            cropped_images (list of torch.Tensor): List of cropped images.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        _, _, H, W = image.shape
        device = image.device
        dtype = image.dtype

        cropped_images = []

        for config in cropping_configs:
            x1, y1, crop_width, crop_height = config

            # Compute the affine transformation matrix
            theta = self._compute_affine_matrix(x1, y1, crop_width, crop_height, W, H)
            theta = theta.to(device=device, dtype=dtype).unsqueeze(0)  # Shape (1, 2, 3)

            # Generate grid and sample
            grid = F.affine_grid(theta, torch.Size((1, image.shape[1], crop_height, crop_width)), align_corners=False)
            cropped_image = F.grid_sample(image, grid, align_corners=False)

            cropped_images.append(cropped_image.squeeze(0))  # Remove batch dimension

        return cropped_images

    def _compute_affine_matrix(self, x1, y1, crop_width, crop_height, W, H):
        """
        Compute the affine transformation matrix for cropping.

        Args:
            x1 (int): Top-left x-coordinate of the crop.
            y1 (int): Top-left y-coordinate of the crop.
            crop_width (int): Width of the crop.
            crop_height (int): Height of the crop.
            W (int): Width of the original image.
            H (int): Height of the original image.

        Returns:
            theta (torch.Tensor): Affine transformation matrix of shape (2, 3).
        """
        # Compute scale factors
        scale_x = crop_width / W
        scale_y = crop_height / H

        # Compute translation factors
        tx = ((2 * x1 + crop_width - W) / W)
        ty = ((2 * y1 + crop_height - H) / H)

        # Affine matrix
        theta = torch.tensor([
            [scale_x, 0, tx],
            [0, scale_y, ty]
        ])
        return theta

if __name__ == '__main__':

    cropper = ImageCropper()

    # Define parameters
    num_augs = 5
    image_shape = (256, 512)
    aspect_ratio_range = (0.8, 1.25)
    min_size = 0.3
    max_size = 0.5

    # Generate cropping configurations
    cropping_configs = cropper.get_cropping_configs(num_augs, image_shape, aspect_ratio_range, min_size, max_size)
    # Load or create an image tensor (3, H, W)
    image = torch.randn(3, 256, 256)

    # Apply cropping configurations
    cropped_images = cropper.apply_cropping_configs(image, cropping_configs)

    # Each element in cropped_images is a cropped image tensor
    for idx, cropped_img in enumerate(cropped_images):
        print(f"Cropped image {idx} shape: {cropped_img.shape}")