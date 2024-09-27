# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Code in this file is adapted from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# This code is modified version of one of ildoonet, for randaugmentation of fixmatch.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image



def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Blur(img, _):
    kernel_size = int(random.random() * 4.95)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    return img.filter(PIL.ImageFilter.GaussianBlur(kernel_size))

def Brightness(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v):
    assert v >= 0.0
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)

def Sharpness(img, v):  # [0.1,1.9]
    assert v >= 0.0
    return PIL.ImageEnhance.Sharpness(img).enhance(v)



def augment_list_no_geometric():
    l = [
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        # (Color, 0.05, 0.95),
        (Contrast, 0.25, 0.95),
        # (Equalize, 0, 1),
        (Posterize, 4, 8),
        (Sharpness, 0.05, 0.95),
        (Blur, 0, 1),
    ]
    return l


class RandAugment:
    def __init__(self, n, m, exclude_color_aug=False):
        self.n = n
        self.m = m  # [0, 30] in fixmatch, deprecated.

        self.augment_list = augment_list_no_geometric()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)
        return img


