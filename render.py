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
import gc
import os
from os import makedirs
import json
import time

import torch
import torchvision
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim
from lpipsPyTorch import lpips
import joblib

from scene import Scene, init_model, im_reader
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.loss_utils import ssim
from random import choice
from utils.sh_utils import RGB2SH
import cv2
import numpy as np
from scene.cameras import Cameravideo

def render_set(model_path, name, iteration, scene, gaussians, pipeline, background, inverval=1, near=0.2, far=100.0, save_img=False, measure_time=False):
    gts_path = os.path.join(model_path, name, "itrs_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "itrs_{}".format(iteration), "renders")

    if save_img:
        makedirs(gts_path, exist_ok=True)
        makedirs(render_path, exist_ok=True)
    
    if name == "train":
        viewpoint_stack, images = scene.getTrainCameras(return_as='generator', shuffle=False)
        viewpoint_stack = viewpoint_stack.copy()
        
    else:
        viewpoint_stack, images = scene.getTestCameras(return_as='generator', shuffle=False)
        viewpoint_stack = viewpoint_stack.copy()
        
    idx = 0
    count = 0
    psnr_sum = 0
    
    psnrs = []
    ssims = []
    skssims = []
    skssims2 = []
    lpipss = []
    lpipssvggs = []
    image_names = []
    times = []
    
    render_mode = 0
    for i in tqdm(range(len(viewpoint_stack)), desc="Rendering"):
        cam = viewpoint_stack.pop(0)
        gt = next(images).cuda()

        if i % inverval == 0:
            rendering_dict = render(cam, gaussians, pipeline, background, mode=render_mode, near=near, far=far)
            rendering = rendering_dict["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)
            
            img_name = cam.image_name
            
            if save_img:
                torchvision.utils.save_image(rendering, os.path.join(render_path, img_name))
            psnrs.append(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)))
            ssims.append(ssim(rendering.unsqueeze(0), gt.unsqueeze(0))) 
            skssims.append(sk_ssim(rendering.detach().cpu().numpy(), gt.detach().cpu().numpy(), data_range=1, multichannel=True, channel_axis=0)) 
            skssims2.append(sk_ssim(rendering.detach().cpu().numpy(), gt.detach().cpu().numpy(), data_range=2, multichannel=True, channel_axis=0)) 
            lpipss.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='alex')) #
            lpipssvggs.append(lpips(rendering.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))
            
            image_names.append(img_name)

            psnr_sum += psnr(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().detach().item()
            count += 1
            
            last_gt = gt
            last_rendering = rendering
        
    # start timing
    if measure_time:
        for _ in range(5):
            for idx in range(500):
                st = time.time()
                rendering_dict = render(cam, gaussians, pipeline, background, mode=render_mode, near=near, far=far)
                if idx > 100: #warm up
                    times.append(time.time() - st)
        
    mean_results = {
        "SSIM": torch.tensor(ssims).mean().item(),
        "SKSSIM": torch.tensor(skssims).mean().item(),
        "SKSSIM2": torch.tensor(skssims2).mean().item(),
        "PSNR": torch.tensor(psnrs).mean().item(),
        "LPIPS": torch.tensor(lpipss).mean().item(),
        "LPIPSVGG": torch.tensor(lpipssvggs).mean().item(),
        "num_splats": gaussians._xyz.shape[0],
        "num_dynamic": (gaussians._level[:gaussians._xyz.shape[0]] != 0).sum().item(),
        "num_group": gaussians._xyz_motion.shape[0],
        }
    
    if measure_time:
        mean_results["times"] = torch.tensor(times).mean().item()
        mean_results["FPS"] = 1 / torch.tensor(times).mean().item()
    
    with open(model_path + "/" + "mean_metrics.json", 'w') as fp:
        json.dump(mean_results, fp, indent=True)

    per_view_results = {
        "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
        "SKSSIM": {name: ssim for ssim, name in zip(torch.tensor(skssims).tolist(), image_names)},
        "SKSSIM2": {name: ssim for ssim, name in zip(torch.tensor(skssims2).tolist(), image_names)},
        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
        "LPIPSVGG": {name: lpipssvgg for lpipssvgg, name in zip(torch.tensor(lpipssvggs).tolist(), image_names)},
    }

    with open(model_path + "/" + "all_metrics.json", 'w') as fp:
        json.dump(per_view_results, fp, indent=True)

    print_msg = f"Set {name}, PSNR: {psnr_sum / count}, Count: {count}, SSIM: {mean_results['SSIM']}, SKSSIM: {mean_results['SKSSIM']}, SKSSIM2: {mean_results['SKSSIM2']}, LPIPS: {mean_results['LPIPS']}, LPIPSVGG: {mean_results['LPIPSVGG']}"
    if measure_time:
        print_msg += f", Times: {mean_results['times']}, FPS: {mean_results['FPS']}"
    print(print_msg)
    

def render_sets(dataset : ModelParams, iteration : int, opt : OptimizationParams, pipeline : PipelineParams, skip_train : bool, train_inverval : int, skip_test : bool, save_img : bool):
    with torch.no_grad():
        gaussians = init_model(dataset.model, dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, opt=opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        splits = [] if skip_train else ["train"]
        splits += [] if skip_test else ["test"]
        for split in splits:
            render_set(dataset.model_path, split, scene.loaded_iter, scene, gaussians, pipeline, background, train_inverval, near=dataset.near, far=dataset.far, save_img=save_img)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--train_inverval", default=1, type=int)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    args = get_combined_args(parser)
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, opt.extract(args), pipeline.extract(args), args.skip_train, args.train_inverval, args.skip_test, args.save_img)