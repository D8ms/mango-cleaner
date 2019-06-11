import numpy as np
import os
import cv2
from PIL import Image
import random

def rand_crop(img, length=256, w_margin=0, h_margin=0):
    height, width = img.shape
    w = random.randint(w_margin, width - length - w_margin)
    h = random.randint(h_margin, height - length - h_margin)
    cropped = img[h:h+length, w:w+length, None]
    return cropped


def rand_bbox(height=256, width=256, length=128):
    t = random.randint(0, height - length)
    l = random.randint(0, width - length)
    #tlhw
    return (t, l, length, length)

def bbox_to_mask(bbox, length=256):
    mask = np.zeros((1, length, length, 1), np.float32)
    dh = np.random.randint(17)
    dw = np.random.randint(17)
    mask[:, bbox[0] + dh : bbox[0] + bbox[2] - dh, bbox[1] + dw : bbox[1] + bbox[3] - dw, :] = 1
    return mask

def simple_discounting_mask(length=128):
    gamma = 0.9
    mask_values = np.ones((length, length))
    for i in range(length):
        for j in range(length):
            mask_values[i, j] = max(
                gamma**min(i, length - i),
                gamma**min(j, length - j))
    mask_values = np.expand_dims(mask_values, 0)
    mask_values = np.expand_dims(mask_values, 3)
    return mask_values

def rand_crop_avoid_mask(height, width, raw, mask):
    #crop section of the image that is not masked
    pass

def gen_text_mask(width, height, scale=1):
    #create a block of kanji (maybe hiragana as well?) text
    pass

def apply_text_mask(image, mask):
    pass
