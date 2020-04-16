import numpy as np
import math

def extract_brain(image_voxel, brain_voxel):
    return image_voxel * brain_voxel

def weighted_avg_and_std(values, weights=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, math.sqrt(variance)

def normalize(image_voxel, weights=None):
    ave, std = weighted_avg_and_std(image_voxel, weights=weights)
    image_voxel = (image_voxel - ave) / std
    return image_voxel


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