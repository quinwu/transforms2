from __future__ import division
import torch
import math
import sys
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

import torchvision.transforms.functional as TF


class Compose2(object):
    def __init__ (self,transforms):
        self.transforms = transforms
        self.PIL2Numpy = False

    def __call__(self,img,mask):
    
        if isinstance (img,np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="F")
            self.PIL2Numpy = True
        
        assert img.size == mask.size 

        print (type(self.transforms))

        for t in self.transforms:
            img, mask = t(img, mask)
        
        if self.PIL2Numpy:
            img,mask = np.array(img), np.array(mask)
        
        return img, mask

class Resize2(object):

    def __init__(self,size,interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img, mask):
        return TF.resize(img, self.size, self.interpolation), TF.resize(mask, self.size, self.interpolation)

class RandomHorizontalFlip2(object):
    
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class RandomVerticalFlip2(object):
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask

class RandomRotation(object):

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, mask):
        
        angle = self.get_params(self.degrees)
        return TF.rotate(img, angle, self.resample, self.expand, self.center), TF.rotate(mask, angle, self.resample, self.expand, self.center)

