#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from PIL import Image
import os

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    return 20 * torch.log10(1.0 / torch.sqrt(mse(img1, img2)))

def psnrmask(img1, img2):
    mask = img2 > 0.0
    validmask = torch.sum(img2[:, :, :], dim=1) > 0.01
    validmask = validmask.repeat(3, 1, 1)#.float()
    validmask = validmask.view(img1.shape[0], -1)
    
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1)[:, validmask[0,:]].mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def PILtoTorch(pil_image, resolution=None, device=None):
    if resolution is not None:
        resized_image_PIL = resizePIL(pil_image, resolution)
    else:
        resized_image_PIL = pil_image
    resized_image = np.array(resized_image_PIL)
    resized_image = torch.from_numpy(resized_image)
    if device is not None:
        resized_image = resized_image.to(device)
    resized_image = resized_image / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def resizePIL(pil_image, resolution):
    if pil_image.size[0] != resolution[0] or pil_image.size[1] != resolution[1]:
        resized_image_PIL = pil_image.resize(resolution)
    else:
        resized_image_PIL = pil_image
    return resized_image_PIL

def load_image_like_train(image_path):
    loaded_image = Image.open(image_path)
    resized_image_rgb = PILtoTorch(loaded_image)
    
    gt_image = resized_image_rgb[:3, ...]
    image_name = os.path.basename(image_path).split(".")[0]
    if "camera_" not in image_name:
        original_image = gt_image.clamp(0.0, 1.0)
    else:
        original_image = gt_image.clamp(0.0, 1.0).half()
        
    return original_image
