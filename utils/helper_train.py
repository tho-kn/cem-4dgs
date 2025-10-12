#
# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import torch
import numpy as np
import torch
from simple_knn._C import distCUDA2
import os 
import json 
import cv2
from scripts.pre_immersive_distorted import SCALEDICT

# fisheye mapper cache
rectify_mappers = {}

def undistortimage(imagename, datasetpath,data, clip_value=True):
    video = os.path.dirname(datasetpath) # upper folder 
    with open(os.path.join(video + "/models.json"), "r") as f:
                meta = json.load(f)

    for idx , camera in enumerate(meta):
        folder = camera['name'] # camera_0001
        if folder != imagename:
             continue
        view = camera
        
        if folder in rectify_mappers:
            map1, map2 = rectify_mappers[folder]
        else:
            intrinsics = np.array([[view['focal_length'], 0.0, view['principal_point'][0]],
                                [0.0, view['focal_length'], view['principal_point'][1]],
                                [0.0, 0.0, 1.0]])
            dis_cef = np.zeros((4))

            dis_cef[:2] = np.array(view['radial_distortion'])[:2]
            # print("done one camera")
            map1, map2 = None, None
            sequencename = os.path.basename(video)
            focalscale = SCALEDICT[sequencename]
    
            h, w = data.shape[:2]


            image_size = (w, h)
            knew = np.zeros((3, 3), dtype=np.float32)


            knew[0,0] = focalscale * intrinsics[0,0]
            knew[1,1] = focalscale * intrinsics[1,1]
            knew[0,2] =  view['principal_point'][0] # cx fixed half of the width
            knew[1,2] =  view['principal_point'][1] #
            knew[2,2] =  1.0
            
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(intrinsics, dis_cef, R=None, P=knew, size=(w, h), m1type=cv2.CV_32FC1)
            
            rectify_mappers[folder] = (map1, map2)

        undistorted_image = cv2.remap(data, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        if clip_value:
            undistorted_image = undistorted_image.clip(0,255.0) 
        return undistorted_image
    
def undistortcudaimage(image, dataset_source, viewpoint_cam):
    image_pm = image.clone().permute(1,2,0) * 255.0
    image_pm = image_pm.clip(0, 255.0)
    imagenumpy= image_pm.cpu().numpy()
    imagex2 =  cv2.resize(imagenumpy, dsize=(viewpoint_cam.image_width, viewpoint_cam.image_height), interpolation=cv2.INTER_CUBIC)
    imagex2ud = undistortimage(viewpoint_cam.image_name, dataset_source, imagex2)
    image = torch.from_numpy(imagex2ud).cuda().permute(2,0,1) / 255.0
    return image

def trbfunction(x, plateau=None):
    if plateau is None:
        return torch.exp(-1*x.pow(2))
    else:
        return torch.exp(-1*((x*x).pow(plateau)))
    
def trbfinverse(y, plateau):
    inter = -torch.log(y)
    return inter.pow(1.0/2.0/plateau)

# We want plateau where inverse 0.1 / inverse 0.9 = (r+1) / r
def trbfinverse10(plateau):
    inv2pl = 1.0/2.0/plateau
    return (-torch.log(torch.tensor(0.1))).to(plateau.device).pow(inv2pl)

def trbfinverse90(plateau):
    inv2pl = 1.0/2.0/plateau
    return (-torch.log(torch.tensor(0.9))).to(plateau.device).pow(inv2pl)

# (log0.1/log0.9)^(1/2.0/plateau) = 1 + 1 / r
# log(log0.1/log0.9) * (1/2.0/plateau) = log(1 + 1 / r)
# log(log0.1/log0.9) / 2 / log(1 + 1 / r) = plateau

# trbfinverse90(plateau) = r / duration / scale
