import importlib
import io
import logging
import os
import shutil
import sys
import uuid

import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.sparse as sparse
import torch
from PIL import Image
from sklearn.decomposition import PCA

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)

def z_pad(image,target_d):
    depth, height, width = image.shape
    if depth == target_d: return image
    if depth > target_d:
        target = depth - target_d
        if target % 2 == 0:
            n_pad_up, n_pad_down = int(target / 2), int(target / 2)
        else:
            n_pad_up, n_pad_down = int(target // 2) + 1, int(target // 2)
        image = image[n_pad_up:n_slices-n_pad_down]
    else:
        target = target_d - depth
        if target % 2 == 0:
            n_pad_up, n_pad_down = int(target / 2), int(target / 2)
        else:
            n_pad_up, n_pad_down = int(target // 2) + 1, int(target // 2)
        up_voxel = np.zeros((n_pad_up, height, width))
        down_voxel = np.zeros((n_pad_down, height, width))
        image = np.concatenate([up_voxel, image, down_voxel])
    return image

def xy_pad(image,target_h,target_w):
    depth, height, width = image.shape
    diff_h = int(abs((height - target_h)/2))
    diff_w = int(abs((width - target_w)/2))
    if height > target_h:
        if height % 2 == 0:
            image = image[:,diff_h:height-diff_h,:]
        else:
            image = image[:,diff_h+1:height-diff_h,:]
    else:
        if height % 2 == 0:
            image = np.pad(image, ((0,0),(diff_h,diff_h),(0,0)), 'constant')
        else:
            image = np.pad(image, ((0,0),(diff_h+1,diff_h),(0,0)), 'constant')
    if width > target_w:
        if width % 2 == 0:
            image = image[:,:,diff_w:width-diff_w]
        else:
            image = image[:,:,diff_w+1:width-diff_w]
    else:
        if width % 2 == 0:
            image = np.pad(image, ((0,0),(0,0),(diff_w,diff_w)), 'constant')
        else:
            image = np.pad(image, ((0,0),(0,0),(diff_w+1,diff_w)), 'constant')
    return image
def xyz_pad(image,target_d,target_h,target_w):
    xy_padded_image_array = xy_pad(image,target_h,target_w)
    xyz_padded_image_array = z_pad(xy_padded_image_array,target_d)
    return xyz_padded_image_array

def read_and_crop(file,target_d=None,target_h=None,target_w=None):
    img = np.load(file)
    d, h, w = img.shape
    if (target_d == None):
        target_d = d
    if (target_h == None):
        target_h = h
    if (target_w == None):
        target_w = w
    img = xyz_pad(np.load(file),target_d,target_h,target_w)
    return img