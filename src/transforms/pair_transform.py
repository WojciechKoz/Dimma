from typing import Tuple

import albumentations as A
import numpy as np
import cv2
import torch
from einops import rearrange


def retinex_decomposition(img):
    R = img / (img.sum(axis=2, keepdims=True) + 1e-6)
    L = (img / (3 * R + 1e-6)).max(axis=2)
    return R, L


class PairedTransformForDimma:
    """
    Uses similar pipeline as MDNTransform but doesn't dim the photo.
    Instead takes already dark image with corresponding
    light one and augment them to fit Dimma model.
    """

    def __init__(self, flip_prob=0.5, crop_size=None, test=False):
        if crop_size is None:
            crop_size = 256

        self.flip_prob = flip_prob
        self.crop_size = (crop_size, crop_size)
        self.test = test

    def common_horizontal_flip(
        self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        if np.random.random() < self.flip_prob:
            return np.flip(image, 1), np.flip(target, 1)
        return image, target

    def common_random_crop(
        self, image: np.array, target: np.array
    ) -> Tuple[np.array, np.array]:
        width = image.shape[0]
        height = image.shape[1]

        start_x = (
            np.random.randint(low=0, high=(width - self.crop_size[0]) + 1)
            if width > self.crop_size[0]
            else 0
        )
        start_y = (
            np.random.randint(low=0, high=(height - self.crop_size[1]) + 1)
            if height > self.crop_size[1]
            else 0
        )

        crop_slice = np.s_[
            start_x : start_x + self.crop_size[0],
            start_y : start_y + self.crop_size[1],
            :,
        ]

        return image[crop_slice], target[crop_slice]

    def __call__(self, image, target):
        """
        :param image: dark image (input)
        :param target: light image (ground truth)
        :return: dict of prepared dark image, target image and lightnesses of both
        """
        if not self.test:
            dark, light = self.common_horizontal_flip(image, target)
            dark, light = self.common_random_crop(dark, light)
        else:
            dark, light = image, target

        # get color map and luminance of image
        R, L = retinex_decomposition(dark / 255.0)

        # get luminance of target and calc mean
        R_target, L_target = retinex_decomposition(light / 255.0)

        # histogram equalization for image
        hist_eq = A.augmentations.functional.equalize(dark) / 255.0

        # concatenate and normalize all channels
        dark = np.concatenate([dark / 255.0, hist_eq, R, L[:, :, None]], axis=2).transpose(2, 0, 1)

        dark = torch.from_numpy(dark).float()
        source_lightness = torch.tensor(L.mean()).float()
        target_lightness = torch.tensor(L_target.mean()).float()

        # concatenate R_target and L_target
        light = torch.from_numpy(light.transpose(2, 0, 1).copy()).float() / 255.0

        # dark = 2 * dark - 1
        # light = 2 * light - 1

        return {
            "image": dark,
            "target": light,
            "source_lightness": source_lightness,
            "target_lightness": target_lightness,
        }
