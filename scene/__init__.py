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

import os
import random
import json
import copy
import math
from itertools import compress
from collections import defaultdict

from PIL import Image, ImageFile
import torch
from joblib import delayed, Parallel
import numpy as np

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.v_gaussian_model import VGaussianModel
from utils.system_utils import searchForMaxIteration
from scene.cameras import cameraList_from_camInfosVideo2, camera_to_JSON
from utils.general_utils import PILtoTorch

def getfisheyemapper(folder, cameraname):
    distoritonflowpath = os.path.join(folder, cameraname + "_half.npy")
    distoritonflow = np.load(distoritonflowpath)
    distoritonflow = torch.from_numpy(distoritonflow).unsqueeze(0).float().cuda()
    return distoritonflow
            
def im_reader(path, resolution, im_scale):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return (PILtoTorch(Image.open(path), resolution)[:3, ...] / im_scale).clamp(0, 1)

class Scene:
    # gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians, load_iteration=None, shuffle=False, resolution_scales=[1.0], load_cameras=True, opt=None, use_timepad=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.lazy_loader = args.lazy_loader
        self.loaded_iter = None
        self.gaussians = gaussians
        self.sample_idx = 0
        self.test_sample_idx = 0
        self.min_timestamp = 0
        self.sample_len = 0
        self.last_sample_len = 0
        self.start_timestamp = args.start_timestamp
        self.end_timestamp = args.end_timestamp

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        loader = args.loader
        if loader == "colmap" or loader == "colmapvalid": # colmapvalid only for testing
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args)
        elif loader == "technicolor" or loader == "technicolorvalid":
            scene_info = sceneLoadTypeCallbacks["Technicolor"](args.source_path, args.images, args.eval, args)
        elif loader == "neural3dvideo":
            scene_info = sceneLoadTypeCallbacks["Neural3DVideo"](args.source_path, args.images, args.eval, args)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        if load_cameras:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                
                if loader in ["colmapvalid", "colmap", "technicolorvalid", "technicolor", "neural3dvideo"]:     
                    self.train_cameras[resolution_scale] = cameraList_from_camInfosVideo2(scene_info.train_cameras, resolution_scale, args)
                    self.train_cameras[resolution_scale] = sorted(self.train_cameras[resolution_scale], key=lambda x: (x.timestamp, x.colmap_id))
                    
                print("Loading Test Cameras")
                if loader in ["colmapvalid", "colmap", "technicolorvalid", "technicolor", "neural3dvideo"]:  
                    self.test_cameras[resolution_scale] = cameraList_from_camInfosVideo2(scene_info.test_cameras, resolution_scale, args)
                    self.test_cameras[resolution_scale] = sorted(self.test_cameras[resolution_scale], key=lambda x: (x.timestamp, x.colmap_id))
            
            max_duration = args.duration
            for resolution_scale in resolution_scales:
                unique_times = set([i.timestamp for i in self.train_cameras[resolution_scale]])
                unique_cids = set([i.colmap_id for i in self.train_cameras[resolution_scale]])
                self.cam_num = len(unique_cids)
                max_duration = max(max_duration, len(unique_times))
                    
                unique_test_times = set([i.timestamp for i in self.test_cameras[resolution_scale]])
                max_duration = max(max_duration, len(unique_test_times))
            
            if args.duration < 0:
                args.duration = max_duration     
                   
            if use_timepad:
                if gaussians.time_pad_type == 1:
                    for resolution_scale in resolution_scales:
                        cid_len = len(unique_cids)
                        prefix_pad = copy.deepcopy(self.train_cameras[resolution_scale][cid_len:cid_len*(gaussians.time_pad+1)])
                        min_timestamp = min(unique_times)
                        
                        for cam in prefix_pad:
                            cam.timestamp = 2 * min_timestamp - cam.timestamp
                        postfix_pad = copy.deepcopy(self.train_cameras[resolution_scale][-cid_len*(gaussians.time_pad+1):-cid_len])
                        max_timestamp = max(unique_times)
                        for cam in postfix_pad:
                            cam.timestamp = 2 * max_timestamp - cam.timestamp
                            
                        self.train_cameras[resolution_scale] = prefix_pad + self.train_cameras[resolution_scale] + postfix_pad
                
                elif gaussians.time_pad_type == 2:
                    for resolution_scale in resolution_scales:
                        cid_len = len(unique_cids)
                        first_frames = copy.deepcopy(self.train_cameras[resolution_scale][:cid_len])
                        prefix_pad = []
                        
                        for i in range(gaussians.time_pad+1):
                            new_frames = copy.deepcopy(first_frames)
                            for cam in new_frames:
                                cam.timestamp = cam.timestamp - i
                            prefix_pad = new_frames + prefix_pad
                            
                        last_frames = copy.deepcopy(self.train_cameras[resolution_scale][-cid_len:])
                        postfix_pad = []
                        for i in range(gaussians.time_pad+1):
                            new_frames = copy.deepcopy(last_frames)
                            for cam in new_frames:
                                cam.timestamp = cam.timestamp + i
                            postfix_pad = postfix_pad + new_frames
                            
                        self.train_cameras[resolution_scale] = prefix_pad + self.train_cameras[resolution_scale] + postfix_pad
                
                self.train_cameras[resolution_scale] = sorted(self.train_cameras[resolution_scale], key=lambda x: (x.timestamp, x.colmap_id))
                 
        if self.loaded_iter:
            if os.path.exists(os.path.join(self.model_path, "chkpnt{}.pth".format(self.loaded_iter))) and not opt is None:
                print("Loading from checkpoint")
                model_args = torch.load(os.path.join(self.model_path, "chkpnt{}.pth".format(self.loaded_iter)), weights_only=False)
                self.gaussians.restore(model_args[0], opt)
                self.set_sampling_len(gaussians.duration)
            else:
                print("Loading from ply")
                gaussians.duration = args.duration
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
                self.set_sampling_len(gaussians.duration)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
    def getTCams(self, scale=1.0, split="train"):
        if split == "train":
            return list(compress(self.train_cameras[scale], self.samplelist))
        elif split == "test":
            return list(compress(self.test_cameras[scale], self.test_samplelist))
        else:
            raise ValueError("Invalid split value. Must be 'train' or 'test'.")
    
    def getNextFrame(self, cam, scale=1.0, split="train", step=1):
        colmap_id = cam.colmap_id
        next_timestamp = cam.timestamp + step
        next_cam = next((i for i in self.getTCams(scale, split) if i.colmap_id == colmap_id and i.timestamp == next_timestamp), None)
        return next_cam
    
    def getPrevFrame(self, cam, scale=1.0, split="train", step=1):
        colmap_id = cam.colmap_id
        prev_timestamp = cam.timestamp - step
        prev_cam = next((i for i in self.getTCams(scale, split) if i.colmap_id == colmap_id and i.timestamp == prev_timestamp), None)
        return prev_cam
    
    def getColmapIds(self, split="train"):
        return [i.colmap_id for i in self.getTCams(split=split)]
    
    def getTimestamps(self, split="train"):
        return [i.timestamp for i in self.getTCams(split=split)]

    def getTrainCameras(self, scale=1.0, shuffle=True, return_as="generator", n_job=2, return_path=False, get_img=True, job_batch_size=2):
        if self.lazy_loader:
            t_cams = self.getTCams(scale, "train")
            t_imgs = [(i.image_path, i.resolution, i.im_scale) for i in t_cams]
            
            if shuffle:
                if job_batch_size == 1:
                    temp = list(zip(t_cams, t_imgs))
                    random.shuffle(temp)
                    res1, res2 = zip(*temp)
                    t_cams, t_imgs = list(res1), list(res2)
                else:
                    # Group images by timestamp
                    groups = defaultdict(list)
                    for cam, img in zip(t_cams, t_imgs):
                        groups[cam.timestamp].append((cam, img))
                    
                    # Shuffle within each group and by batch size
                    grouped_batches = []
                    for group in groups.values():
                        random.shuffle(group)
                        # Discard remainder views that don't fit into a full batch
                        batch_size = job_batch_size  # or another defined batch size
                        num_full_batches = len(group) // batch_size
                        for i in range(num_full_batches):
                            batch = group[i * batch_size:(i + 1) * batch_size]
                            grouped_batches.append(batch)
                    
                    # Shuffle the batches
                    random.shuffle(grouped_batches)
                    
                    # Flatten the list of batches back to separate lists
                    t_cams, t_imgs = zip(*[item for batch in grouped_batches for item in batch])
                    t_cams, t_imgs = list(t_cams), list(t_imgs)
            
            if return_path:
                return t_cams, t_imgs
                        
            return  t_cams, \
                    Parallel(n_jobs=n_job, pre_dispatch='n_jobs', return_as=return_as, batch_size=job_batch_size, backend='loky')(delayed(im_reader)(path, resolution, im_scale) for path, resolution, im_scale in t_imgs) if get_img else None
        else:
            t_cams = self.getTCams(scale, "train")
            if return_path:
                t_imgs = [(i.image_path, i.resolution) for i in t_cams]
            else:
                t_imgs = [i.image for i in t_cams]
            
            if shuffle:
                if job_batch_size == 1:
                    temp = list(zip(t_cams, t_imgs))
                    random.shuffle(temp)
                    res1, res2 = zip(*temp)
                    t_cams, t_imgs = list(res1), list(res2)
                else:
                    # Group images by timestamp
                    groups = defaultdict(list)
                    for cam, img in zip(t_cams, t_imgs):
                        groups[cam.timestamp].append((cam, img))
                    
                    # Shuffle within each group and by batch size
                    grouped_batches = []
                    for group in groups.values():
                        random.shuffle(group)
                        # Discard remainder views that don't fit into a full batch
                        batch_size = job_batch_size  # or another defined batch size
                        num_full_batches = len(group) // batch_size
                        for i in range(num_full_batches):
                            batch = group[i * batch_size:(i + 1) * batch_size]
                            grouped_batches.append(batch)
                    
                    # Shuffle the batches
                    random.shuffle(grouped_batches)
                    
                    # Flatten the list of batches back to separate lists
                    t_cams, t_imgs = zip(*[item for batch in grouped_batches for item in batch])
                    t_cams, t_imgs = list(t_cams), list(t_imgs)
                
            if return_path:
                return t_cams, t_imgs
            
            if return_as == "list":
                return t_cams, t_imgs
            else:
                def img_iterator():
                    for  img in t_imgs:
                        yield img
                return t_cams, img_iterator()
            
    def getTestCameras(self, scale=1.0, shuffle=True, return_as="generator", n_job=2, return_path=False, get_img=True, job_batch_size=2):
        if self.lazy_loader:
            t_cams = self.getTCams(scale, "test")
            t_imgs = [(i.image_path, i.resolution, i.im_scale) for i in t_cams]
            
            if shuffle:
                temp = list(zip(t_cams, t_imgs))
                random.shuffle(temp)
                res1, res2 = zip(*temp)
                t_cams, t_imgs = list(res1), list(res2)
                
            if return_path:
                return t_cams, t_imgs

            return  t_cams, \
                    Parallel(n_jobs=n_job, pre_dispatch='n_jobs', return_as=return_as, batch_size=job_batch_size, backend='loky')(delayed(im_reader)(path, resolution, im_scale) for path, resolution, im_scale in t_imgs) if get_img else None
        else:
            t_cams = self.getTCams(scale, "test")
            if return_path:
                t_imgs = [(i.image_path, i.resolution) for i in t_cams]
            else:
                t_imgs = [i.image for i in t_cams]
                        
            if shuffle:
                temp = list(zip(t_cams, t_imgs))
                random.shuffle(temp)
                res1, res2 = zip(*temp)
                t_cams, t_imgs = list(res1), list(res2)
                
            else:
                def img_iterator():
                    for img in t_imgs:
                        yield img

                return t_cams, img_iterator()
    
    def set_sampling_len(self, sample_len, min_timestamp=0, sample_every=1):
        self.samplelist = [i.timestamp <= sample_len and i.timestamp >= min_timestamp and i.timestamp % sample_every == 0 for i in self.train_cameras[1.0]]
        self.test_samplelist = [i.timestamp <= sample_len for i in self.test_cameras[1.0]]
        self.min_timestamp = min_timestamp
        self.last_sample_len = self.sample_len
        self.sample_len = sample_len
        
    def get_last_sample_cameras(self):
        last_cameras = [i for i in self.train_cameras[1.0] if self.last_sample_len <= i.timestamp <= self.sample_len]
        return last_cameras
    
def getmodel(model="virtual"):
    if model == "virtual":
        return VGaussianModel
    else:
        raise NotImplementedError("model {} not implemented".format(model))

def init_model(model, dataset: ModelParams):
    return getmodel(model)(dataset.sh_degree, dataset.start_duration, dataset.time_interval, dataset.time_pad, 
                                interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type, 
                                time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, kernel_size=dataset.kernel_size,
                                connectivity_threshold=dataset.connectivity_threshold)