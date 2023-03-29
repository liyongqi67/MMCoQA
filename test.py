# coding=UTF-8
import json
from tqdm import tqdm
from random import choice
import numpy as np
import random
import pickle
import csv
import uuid
from PIL import Image
import cv2
import torch
random.seed(67)

def image_transform(path):
    MAX_SIZE = 1333
    MIN_SIZE = 800

    img = Image.open(path).convert("RGB")
    im = np.array(img).astype(np.float32)
    # IndexError: too many indices for array, grayscale images
    if len(im.shape) < 3:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    im = im[:, :, ::-1]
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_height = im_shape[0]
    im_width = im_shape[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # Scale based on minimum size
    im_scale = MIN_SIZE / im_size_min

    # Prevent the biggest axis from being more than max_size
    # If bigger, scale it down
    if np.round(im_scale * im_size_max) > MAX_SIZE:
        im_scale = self.MAX_SIZE / im_size_max

    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)

    im_info = {"width": im_width, "height": im_height}

    return img, im_scale, im_info


img, im_scale, im_info=image_transform("/home/share/project/MMCoQA/data/MMCoQA_data/final_data/multimodal_evidence_collection/images/final_dataset_images/35b31d9b4f723f806fd32662ef29edf7.jpg")

print('img',img.shape)
print('im_scale',im_scale)
print('im_info',im_info)