# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
3Augment implementation
Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino)
and timm DA(https://github.com/rwightman/pytorch-image-models)
"""
from torchvision import transforms
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms
import random
from PIL import ImageFilter, ImageOps


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
     
class horizontal_flip(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2,activate_pred=False):
        self.p = p
        self.transf = transforms.RandomHorizontalFlip(p=1.0)
 
    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img
        
class ResizeSmall(object):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio.

    Args:
        smaller_size: an integer, that represents a new size of the smaller side of
        an input image.

    Returns:
        A function, that resizes an image and preserves its aspect ratio.

    """

    def __init__(self, smaller_size):
        assert isinstance(smaller_size, (int))
        self.smaller_size = smaller_size

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]  # image should be a tensor of shape (channels, height, width)

        # Figure out the necessary h/w.
        ratio = float(self.smaller_size) / min(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        image = transforms.Resize((new_h, new_w), antialias=True)(image)
        return image
    
def new_data_aug_generator(simple_random_crop,color_jitter,img_size):
    remove_random_resized_crop = simple_random_crop
    primary_tfl = []
    scale=(0.08, 1.0)
    interpolation='bicubic'
    if remove_random_resized_crop:
        primary_tfl = [
            transforms.Resize(img_size, interpolation=3),
            transforms.RandomCrop(img_size, padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(
                img_size, scale=scale, interpolation=interpolation),
            transforms.RandomHorizontalFlip()
        ]

    secondary_tfl = [transforms.RandomChoice([gray_scale(p=1.0),
                                            Solarization(p=1.0),
                                            GaussianBlur(p=1.0)])]
    if color_jitter is not None and not color_jitter==0:
        secondary_tfl.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))

    return primary_tfl+secondary_tfl