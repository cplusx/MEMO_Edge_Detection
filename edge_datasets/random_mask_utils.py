# Adapted from https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/augmentation/masks.py

import math
import hashlib
from enum import Enum

import os
import cv2
import numpy as np
import os
from PIL import Image

import torch


class LinearRamp:
    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part
    

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_len=10, min_width=5, min_times=0, max_times=10, draw_method='random'):


    height, width = shape
    times = np.random.randint(min_times, max_times + 1)
    masks = np.zeros((times, height, width), np.float32)
    for i in range(times):
        if draw_method == 'random':
            this_draw_method = DrawMethod(np.random.choice(list(DrawMethod)))
        else:
            this_draw_method = DrawMethod(draw_method)
        mask = np.zeros((height, width), np.float32)
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            # length = min_len + np.random.randint(max_len)
            # brush_w = min_width + np.random.randint(max_width)
            length = np.random.randint(min_len, max_len)
            brush_w = np.random.randint(min_width, max_width)
            end_x = np.clip((start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip((start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if this_draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif this_draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif this_draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius, start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
        masks[i] = mask
    return masks # num_masks x height x width


class RandomIrregularMaskEmbedder:
    def __init__(self, max_angle=4, max_len=60, max_width=20, min_len=60, min_width=20, min_times=0, max_times=10, ramp_kwargs=None,
                 draw_method='random'):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_len = min_len
        self.min_width = min_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(**ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, shape, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        if len(shape) == 3:
            this_min_times = min(shape) # (N, H, W) or (H, W, N)
            this_max_times = min(shape)
            min_dim_index = np.argmin(shape)
            if min_dim_index == 0:  # (N, H, W)
                shape = (shape[1], shape[2])
            else:
                shape = (shape[0], shape[1])
        else:
            this_max_times = int(self.min_times + (self.max_times - self.min_times) * coef)
            this_min_times = self.min_times
        return make_random_irregular_mask(
            shape, 
            max_angle=self.max_angle, 
            max_len=cur_max_len, 
            max_width=cur_max_width, 
            min_len=self.min_len, 
            min_width=self.min_width,
            min_times=this_min_times,
            max_times=this_max_times,
            draw_method=self.draw_method
        )

if __name__ == "__main__":
    irregular_kwargs = {'max_angle': 4, 'max_len': 60, 'max_width': 40, 'min_len': 50, 'min_width': 20, 'min_times': 2, 'max_times': 8}
    mask_embedder = RandomIrregularMaskEmbedder(
        **irregular_kwargs
    )
    masks = mask_embedder((256, 256))