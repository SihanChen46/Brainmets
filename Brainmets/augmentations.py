import sys
sys.path.append('../../image_processing/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from scipy import ndimage

def flip(img, mask, axis):
    """flips each slice  along horizontal axis"""
    if axis=='d':
        return np.flip(img, axis=0), np.flip(mask, axis=0)
    if axis=='h':
        return np.flip(img, axis=1), np.flip(mask, axis=1)
    if axis=='w':
        return np.flip(img, axis=2), np.flip(mask, axis=2)

    

    
    
def crop_zoom(image, mask, max_h_zoom=1.2, max_w_zoom=1.2):
    h_zoom = np.random.randint(10,max_h_zoom*10)/10.
    w_zoom = np.random.randint(10,max_w_zoom*10)/10.
    zoom = (1, h_zoom, w_zoom)
    image = ndimage.zoom(image, zoom)
    mask = ndimage.zoom(mask, zoom)
    _, x, y = image.shape
    
    cropx = 256
    cropy = 256
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    

    image = image[:,startx:startx+cropx, starty:starty+cropy].copy()
    mask = (mask[:,startx:startx+cropx, starty:starty+cropy].copy() > 0.5).astype(float)
    return image, mask

def rotate(image_voxel, mask_voxel, angle=15):
    """rotate by +-angle"""
    H, W = mask_voxel.shape[1], mask_voxel.shape[2]
    angle = np.random.randint(-angle, angle, size=1)
    M = cv2.getRotationMatrix2D((H / 2, W / 2), angle, 1)
    image_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in image_voxel])
    mask_voxel = np.array([cv2.warpAffine(slice_, M, (H, W)) for slice_ in mask_voxel])
    return image_voxel, mask_voxel


class Transformer():
    
    def __init__(self,axes, max_zoom_rate, angle):
        self.max_zoom_rate = max_zoom_rate
        self.angle = angle
        self.axes = axes
        
    def transform(self, img, mask):
        trans = np.random.choice([flip, crop_zoom, rotate], 1)
        if trans == flip:
            axis = np.random.choice(self.axes)
            return flip(img, mask, axis)
        if trans == crop_zoom:
            return crop_zoom(img, mask, max_h_zoom=self.max_zoom_rate, max_w_zoom=self.max_zoom_rate)
        if trans ==rotate:
            return rotate(img, mask, angle=self.angle)