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
import math
import os
from scipy.spatial import cKDTree
import networkx as nx
import faiss

import numpy as np
import torch
from torch import nn
from plyfile import PlyData, PlyElement
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert

from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.system_utils import mkdir_p
from utils.interpolations import linear_interp_uniiterval, quat_slerp_interp_uniiterval, time_bigaussian, pchip_interpolate, cube_interpolate, quad_diff_interpolate

class VGaussianModel(torch.nn.Module):
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, duration : int, interval : int, time_pad : int = 1, interp_type='linear', rot_interp_type='slerp', time_pad_type=0, var_pad=3, kernel_size=0.1,
                 connectivity_threshold=0.5, **kwargs):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0).cuda()
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0).cuda()
        self.min_radii2D = torch.empty(0).cuda()
        self.xyz_gradient_accum = torch.empty(0).cuda()
        self.denom = torch.empty(0).cuda()
        self.xyz_error_accum = torch.empty(0).cuda()
        self.xyz_error_min = torch.empty(0).cuda()
        self.xyz_error_min_timestamp = torch.empty(0).cuda()
        self.xyz_ssim_error_accum = torch.empty(0).cuda()
        self.error_denom = torch.empty(0).cuda()
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.var_pad = var_pad
        self.setup_functions()
        
        self.connectivity_threshold = connectivity_threshold
        self.disp_consistency = False # On duration update, update xyz_disp to be consistent with duration
        
        # properties for motion handling
        self.kernel_size = kernel_size
        self.duration = max(duration, 1) # prevent zero div
        self.interval = interval
        self.time_pad = time_pad
        self.time_pad_type = time_pad_type
        self.time_shift = time_pad 
        self.keyframe_num = 0
        self._xyz_disp = torch.empty(0).cuda()
        self._xyz_motion = torch.empty(0).cuda() # [b, t, 3]
        self._opacity_duration_center = torch.empty(0).cuda() # [b, 1] # When static child t opacity is used
        self._opacity_duration_var = torch.empty(0).cuda() # [b, 1]
        self.opacity_degree = 2
        self._rotation_motion = torch.empty(0).cuda() # [b, t, 4]
        
        # For kinematic chain management for relative positioning
        # Assume static, dynamic ordered concatenated array for these
        self._parent = torch.empty(0, dtype=torch.int32, requires_grad=False).cuda()
        self._level = torch.empty(0, dtype=torch.int32, requires_grad=False).cuda()
        
        self.interp_type = interp_type
        # set interpolations
        if interp_type == "linear":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 3)), 
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 3) + 1), delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[...,t_idx, :]
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif interp_type == "cube":
            self.motion_degree = 1 # points, diff
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return cube_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[..., t_idx, :3]
                    return cube_interpolate(y[:, t_idx-1, :3], y[:, t_idx, :3],y[:, t_idx+1, :3], y[:, t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval
            
        elif interp_type == "cubic_diff":
            self.motion_degree = 1 # points, diff
            def interpolation(y, y_d, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y_d1 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y_d2 = torch.gather(y_d, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    return quad_diff_interpolate(y1, y2, y_d1, y_d2, delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[:,t_idx, :]
                    return quad_diff_interpolate(y[:,t_idx], y[:,t_idx+1], y_d[:,t_idx], y_d[:,t_idx+1], delta_t).squeeze(1)
                
        elif interp_type == "pchip":
            self.motion_degree = 1 # points
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    y0 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) - 1)
                    y1 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree))
                    y2 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 1)
                    y3 = torch.gather(y, 1, t_idx.repeat(1, 1, 3 * self.motion_degree) + 2)
                    return pchip_interpolate(y0[..., :3], y1[...,:3],y2[..., :3], y3[...,:3], delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[...,t_idx, :3]
                    return pchip_interpolate(y[...,t_idx-1, :3], y[...,t_idx, :3],y[...,t_idx+1, :3], y[...,t_idx+2, :3], delta_t).squeeze(1)
            self.time_shift += interval
        else:
            raise NotImplementedError
        
        self.interpolator = interpolation
        
        if rot_interp_type == 'lerp':
            def interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return linear_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)), 
                                                torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[...,t_idx, :]
                    return linear_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        elif rot_interp_type == 'slerp':
            # rotation interpolation
            def quat_interpolation(y, t_idx, delta_t):
                if type(t_idx) == torch.Tensor:
                    return quat_slerp_interp_uniiterval(torch.gather(y, 1, t_idx.repeat(1, 1, 4)), 
                                                        torch.gather(y, 1, t_idx.repeat(1, 1, 4) + 1), delta_t).squeeze(1)
                else:
                    if math.isclose(delta_t, 0.0):
                        return y[...,t_idx, :]
                    return quat_slerp_interp_uniiterval(y[...,t_idx, :], y[...,t_idx+1, :], delta_t).squeeze(1)
        else:
            raise NotImplementedError
        self.quat_interpolator = quat_interpolation
        self.error_dict = {}
        
    def get_max_level(self):
        return self._level.max().item()
        
    def get_xyz_at_t(self, t, mode=0):
        # Compute root level
        local_xyz = self.get_local_xyz_at_t(t)
        global_rotation = self.get_rotation_at_t(t, mode=3)
        
        global_xyzs = local_xyz.clone()
        
        # Compute child levels
        child_nodes = (self._level != 0)[:, 0]
        child_xyz = local_xyz[child_nodes]
        parent_nodes = self._parent[child_nodes][:, 0]
        parent_xyz = global_xyzs[parent_nodes]
        global_xyzs[child_nodes] = parent_xyz + quaternion_apply(global_rotation[parent_nodes], child_xyz)

        global_xyzs = global_xyzs.clone()
        global_xyzs[:self._xyz.shape[0]] += self._xyz_disp[:self._xyz.shape[0]] * t / self.duration
        
        if mode == 1:
            return global_xyzs[:self._xyz.shape[0]][(self._level == 0).squeeze(-1)[:self._xyz.shape[0]]]
        elif mode == 2:
            return global_xyzs[:self._xyz.shape[0]][(self._level != 0).squeeze(-1)[:self._xyz.shape[0]]]
        elif mode == 3:
            return global_xyzs
        return global_xyzs[:self._xyz.shape[0]]
    
    def get_local_xyz_at_t(self, t):
        assert t >= -self.time_shift
        assert t <= self.duration + self.time_shift or \
            t <= (self.keyframe_num * self.interval - self.time_shift) and math.isclose(t % 1.0, 0.0)
        if self._xyz_motion.shape[0] == 0:
            return self.get_static_xyz_at_t(t)
        return torch.cat([self.get_static_xyz_at_t(t), self.get_dynamic_xyz_at_t(t)], dim=0).contiguous()
       
    def get_static_xyz_at_t(self, t, mask=None):
        # static points
        if mask is None:
            mask = torch.ones(self._xyz.shape[0], dtype=torch.bool).cuda()
        
        return self._xyz[mask]

    def get_dynamic_xyz_at_t(self, t, mask=None):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval
        
        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
            
        xyz_motion = self._xyz_motion[mask] if mask is not None else self._xyz_motion
        return self.interpolator(xyz_motion, t_idx, delta_t)
    
    def get_rotation_at_t(self, t, mode=0):
        local_rotation = self.get_local_rotation_at_t(t)
        global_rotation = local_rotation.clone()
        
        child_nodes = (self._level != 0)[:, 0]
        child_rotation = local_rotation[child_nodes]
        parent_nodes = self._parent[child_nodes][:, 0]
        parent_rotation = global_rotation[parent_nodes]
        global_rotation[child_nodes] = quaternion_multiply(parent_rotation, child_rotation)
        
        if mode == 1:
            return global_rotation[:self._rotation.shape[0]][(self._level == 0).squeeze(-1)[:self._rotation.shape[0]]]
        elif mode == 2:
            return global_rotation[:self._rotation.shape[0]][(self._level != 0).squeeze(-1)[:self._rotation.shape[0]]]
        elif mode == 3:
            return global_rotation
        return global_rotation[:self._rotation.shape[0]]
       
    def get_local_rotation_at_t(self, t):
        assert t <= self.duration + self.time_shift or \
            t <= (self.keyframe_num * self.interval - self.time_shift) and math.isclose(t % 1.0, 0.0)
        if self._rotation_motion.shape[0] == 0:
            return self.rotation_activation(self._rotation)
        return torch.cat([self.rotation_activation(self._rotation), self.get_dynamic_rotation_at_t(t)], dim=0).contiguous()
    
    def get_dynamic_rotation_at_t(self, t):
        # dynamic points
        t = t + self.time_shift
        t_idx = t // self.interval
        delta_t = (t % self.interval) / self.interval
        
        if type(t) == torch.Tensor:
            t_idx = t_idx.view(-1, 1, 1).long()
            delta_t = delta_t.view(-1, 1, 1)
        else:
            t_idx = int(t_idx)
            
        return self.quat_interpolator(torch.nn.functional.normalize(self._rotation_motion, dim=-1), t_idx, delta_t)
        
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._opacity_duration_center,
            self._opacity_duration_var,
            self.max_radii2D,
            self.min_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.xyz_error_accum,
            self.xyz_error_min,
            self.xyz_error_min_timestamp,
            self.xyz_ssim_error_accum,
            self.error_denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            
            self._parent,
            self._level,
            
            self._xyz_disp,
            self.duration,
            self.interval,
            self.time_shift,
            self.keyframe_num,
            self._xyz_motion,
            self._rotation_motion,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._opacity_duration_center,
        self._opacity_duration_var,
        self.max_radii2D, 
        self.min_radii2D, 
        xyz_gradient_accum, 
        denom,
        xyz_error_accum, 
        xyz_error_min, 
        xyz_error_min_timestamp,
        xyz_ssim_error_accum, 
        error_denom,
        opt_dict, 
        self.spatial_lr_scale,
        
        self._parent,
        self._level,
        
        self._xyz_disp,
        self.duration,
        self.interval,
        self.time_shift,
        self.keyframe_num,
        self._xyz_motion,
        self._rotation_motion,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.xyz_error_accum = xyz_error_accum
        self.xyz_error_min = xyz_error_min
        self.xyz_error_min_timestamp = xyz_error_min_timestamp
        self.xyz_ssim_error_accum = xyz_ssim_error_accum
        self.error_denom = error_denom
        self.optimizer.load_state_dict(opt_dict)
        
    def get_nearest_gaussian(self, xyz, timestamp, opacity_thres=0.01, distance_thres=float('inf')):
        static_xyz = self.get_xyz_at_t(timestamp)[:self._xyz.shape[0]]
        static_opacity = self.get_static_opacity_weight(timestamp) * self.get_opacity[:self._xyz.shape[0]]
        valid_mask = static_opacity.squeeze() > opacity_thres
        if not valid_mask.any():
            return torch.full((xyz.shape[0], 1), -1, dtype=torch.long, device=xyz.device)
        valid_xyz = static_xyz[valid_mask]
        
        data = valid_xyz.detach().cpu().numpy().astype('float32')
        query = xyz.detach().cpu().numpy().astype('float32')

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, data.shape[1])  # L2 distance in 3D
        index.add(data)

        # Search for 1 nearest neighbor (k=1)
        distances, indices = index.search(query, 1)  # distances: (num_points, 1), indices: (num_points, 1)

        # Convert results back to torch tensors on the original device
        min_distances = torch.from_numpy(distances.squeeze(1)).to(xyz.device)
        nearest_indices = torch.from_numpy(indices).to(xyz.device)

        # Apply distance threshold (as in your original code)
        nearest_indices[min_distances > distance_thres] = -1
        return torch.where(
            nearest_indices == -1,
            torch.full_like(nearest_indices, -1),
            torch.arange(self._xyz.shape[0], device=xyz.device)[valid_mask][nearest_indices]
        )

    @property
    def get_static_scaling(self):
        return self.scaling_activation(self._scaling)
    
    def get_scaling(self, mode=0):
        if mode == 1 or mode == 2:
            if mode == 1:
                return self.get_static_scaling[(self._level == 0).squeeze(-1)[:self._xyz.shape[0]]]
            if mode == 2:
                return self.get_static_scaling[(self._level != 0).squeeze(-1)[:self._xyz.shape[0]]]
        return self.get_static_scaling

    def get_features(self, mode=0):
        if mode == 1 or mode == 2:
            features_dc = self._features_dc
            features_rest = self._features_rest
            features = torch.cat((features_dc, features_rest), dim=1)
            if mode == 1:
                return features[(self._level == 0).squeeze(-1)[:self._xyz.shape[0]]]
            elif mode == 2:
                return features[(self._level != 0).squeeze(-1)[:self._xyz.shape[0]]]
            return features
        
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_splat_opacity(self, mode=0):
        if mode == 1 or mode == 2:
            if mode == 1:
                return self.get_opacity[(self._level == 0).squeeze(-1)[:self._xyz.shape[0]]]
            elif mode == 2:
                return self.get_opacity[(self._level != 0).squeeze(-1)[:self._xyz.shape[0]]]
            return self.get_opacity
        return self.get_opacity
       
    def get_static_opacity_weight(self, t, mask=None):
        t = (t + self.time_shift) / self.interval
        if mask is None:
            return time_bigaussian(self._opacity_duration_center, self._opacity_duration_var, t, var_min=self.var_pad/self.interval)
        return time_bigaussian(self._opacity_duration_center[mask], self._opacity_duration_var[mask], t, var_min=self.var_pad/self.interval)
    
    def get_opacity_weight_at_t(self, t, mode=0, mask=None):
        # Static child should inherit parent's opacity weight
        splat_opacity_weight = torch.ones((self._xyz.shape[0], 1), dtype=torch.float, device="cuda")
        
        child_mask = (self._level[:self._xyz.shape[0]] > 0).squeeze(-1)
        if torch.any(child_mask):
            opacity_weight = self.get_static_opacity_weight(t, mask=child_mask)
            splat_opacity_weight[child_mask] = opacity_weight
        
        if mode == 1:
            return splat_opacity_weight[(self._level[:self._xyz.shape[0]] == 0).squeeze(-1)]
        elif mode == 2:
            return splat_opacity_weight[(self._level[:self._xyz.shape[0]] != 0).squeeze(-1)]
        return splat_opacity_weight
    
    # Quality seems to drop after densification is done (confirm it with tensorboard later)
    def get_opacity_at_t(self, t, mode=0):
        assert t <= self.duration + self.time_shift or \
            t <= (self.keyframe_num * self.interval - self.time_shift) and math.isclose(t % 1.0, 0.0)
        splat_opacity = self.get_splat_opacity(mode=mode) * self.get_opacity_weight_at_t(t, mode=mode)
        return splat_opacity.reshape(-1, 1)
    
    def get_covariance_at_t(self, t, scaling_modifier = 1, mode=0):
        # pad not required here
        return self.covariance_activation(self.get_scaling(mode=mode), scaling_modifier, self.get_rotation_at_t(t, mode=mode)).contiguous()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0 # why?

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        opacity_duration_center = torch.zeros((fused_point_cloud.shape[0], 2, 1), device="cuda")
        opacity_duration_center[:, 1] = 1.0
        opacity_duration_var = torch.ones((fused_point_cloud.shape[0], 2, 1), device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._opacity_duration_center = nn.Parameter(opacity_duration_center.requires_grad_(True))
        self._opacity_duration_var = nn.Parameter(opacity_duration_var.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
        self._xyz_disp = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
        
        # All PCD points are at level 0
        self._parent = torch.full((self._xyz.shape[0], 1), -1, device="cuda", dtype=torch.long)
        self._level = torch.zeros((self._xyz.shape[0], 1), device="cuda", dtype=torch.long)
        
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_min = torch.ones((self._xyz.shape[0], 1), device="cuda") * 1000
        self.xyz_error_min_timestamp = torch.ones((self._xyz.shape[0], 1), device="cuda") * -1
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")

        #######################################################################################################
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._opacity_duration_center], 'lr': training_args.opacity_center_lr, "name": "opacity_center"},
            {'params': [self._opacity_duration_var], 'lr': training_args.opacity_var_lr, "name": "opacity_var"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._xyz_disp], 'lr': training_args.disp_lr, "name": "xyz_disp"},
            
            {'params': [self._xyz_motion], 'lr': training_args.dynamic_position_lr_init * self.spatial_lr_scale, "name": "motion_xyz"},
            {'params': [self._rotation_motion], 'lr': training_args.rotation_motion_lr, "name": "motion_rotation"},
        ]

        self.optimizer = torch.optim.RAdam(l, lr=0.001)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.xyz_motion_scheduler_args = get_expon_lr_func(lr_init=training_args.dynamic_position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.dynamic_position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.dynamic_position_lr_delay_mult,
                                                    max_steps=training_args.dynamic_position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif param_group["name"] == "motion_xyz":
                lr = self.xyz_motion_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_static_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._xyz_disp.shape[1]):
            l.append('xyz_disp_{}'.format(i))
        for i in range(self._opacity_duration_center.shape[1]):
            l.append('opacity_c_{}'.format(i))
        for i in range(self._opacity_duration_var.shape[1]):
            l.append('opacity_v_{}'.format(i))
            
        return l
    
    def construct_list_of_dynamic_attributes(self):
        l2 = []
        # attributes for dynamic points.
        for i in range(self._xyz_motion.shape[1]):
            for j in range(self._xyz_motion.shape[2]):
                l2.append('motion_xyz_{}_{}'.format(i, j))
        for i in range(self._rotation_motion.shape[1]):
            for j in range(self._rotation_motion.shape[2]):
                l2.append('motion_rot_{}_{}'.format(i, j))
        
        return l2

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        xyz_disp = self._xyz_disp.detach().cpu().numpy()
        opacity_duration_center = self._opacity_duration_center.detach().flatten(start_dim=1).cpu().numpy()
        opacity_duration_var = self._opacity_duration_var.detach().flatten(start_dim=1).cpu().numpy()
            
        static_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_static_attributes()]

        static_elements = np.empty(xyz.shape[0], dtype=static_dtype_full)
        static_attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, xyz_disp, opacity_duration_center, opacity_duration_var), axis=1)
        static_elements[:] = list(map(tuple, static_attributes))
        s_el = PlyElement.describe(static_elements, 'vertex')
        PlyData([s_el]).write(path)
        
        xyz_motion = self._xyz_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        rotation_motion = self._rotation_motion.detach().flatten(start_dim=1).contiguous().cpu().numpy()

        dynamic_dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_dynamic_attributes()]
        dynamic_elements = np.empty(xyz_motion.shape[0], dtype=dynamic_dtype_full)
        dynamic_attributes = np.concatenate((xyz_motion, rotation_motion), axis=1)
        dynamic_elements[:] = list(map(tuple, dynamic_attributes))
        d_el = PlyElement.describe(dynamic_elements, 'vertex')
        PlyData([d_el]).write(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))
        
        parent = self._parent.detach().cpu().numpy()
        level = self._level.detach().cpu().numpy()
        shared_dtype_full = [(attribute, 'u4') for attribute in ['parent', 'level']]
        shared_elements = np.empty(parent.shape[0], dtype=shared_dtype_full)
        shared_elements[:] = list(map(tuple, np.concatenate((parent, level), axis=1)))
        s_el = PlyElement.describe(shared_elements, 'vertex')
        PlyData([s_el]).write(path.replace('point_cloud.ply', 'shared_point_cloud.ply'))
        
        # Save duration, _xyz_disp depends on it
        duration = torch.tensor([self.duration], dtype=torch.float32)
        duration_element = np.array([(self.duration,)], dtype=[('duration', 'f4')])
        d_el = PlyElement.describe(duration_element, 'duration')
        PlyData([d_el]).write(path.replace('point_cloud.ply', 'duration.ply'))
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.85))
        optimizable_tensors = self.replace_tensor_to_optimizer({"opacity": opacities_new})
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        num_points = xyz.shape[0]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_disp = np.zeros((xyz.shape[0], 3))
        xyz_disp[:, 0] = np.asarray(plydata.elements[0]["xyz_disp_0"])
        xyz_disp[:, 1] = np.asarray(plydata.elements[0]["xyz_disp_1"])
        xyz_disp[:, 2] = np.asarray(plydata.elements[0]["xyz_disp_2"])
        
        opacity_c_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_c_")]
        opacity_c_names = sorted(opacity_c_names, key = lambda x: int(x.split('_')[-1] ))
        opacities_c = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(opacity_c_names):
            opacities_c[:, idx, 0] = np.asarray(plydata.elements[0][attr_name])
        
        opacity_v_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity_v_")]
        opacity_v_names = sorted(opacity_v_names, key = lambda x: int(x.split('_')[-1] ))
        opacities_v = np.zeros((num_points, self.opacity_degree, 1))
        for idx, attr_name in enumerate(opacity_v_names):
            opacities_v[:, idx, 0] = np.asarray(plydata.elements[0][attr_name])
        
        # dynamic points
        plydata_dynamic = PlyData.read(path.replace('point_cloud.ply', 'dynamic_point_cloud.ply'))
        self.keyframe_num = math.ceil((self.duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 4
        
        num_points = plydata_dynamic.elements[0].count
        motion_xyz_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_xyz_")]
        motion_xyz_names = sorted(motion_xyz_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))
        
        motion_xyz = np.zeros((num_points, self.keyframe_num * 3 * self.motion_degree))
        for idx, attr_name in enumerate(motion_xyz_names):
            motion_xyz[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_xyz = motion_xyz.reshape((num_points, self.keyframe_num, 3 * self.motion_degree))
        
        motion_rot_names = [p.name for p in plydata_dynamic.elements[0].properties if p.name.startswith("motion_rot_")]
        motion_rot_names = sorted(motion_rot_names, key = lambda x: ( int(x.split('_')[-2]), int(x.split('_')[-1]) ))
        motion_rots = np.zeros((num_points, self.keyframe_num * 4))
        for idx, attr_name in enumerate(motion_rot_names):
            motion_rots[:, idx] = np.asarray(plydata_dynamic.elements[0][attr_name])
        motion_rots = motion_rots.reshape((num_points, self.keyframe_num, 4))
        
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._xyz_disp = nn.Parameter(torch.tensor(xyz_disp, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_center = nn.Parameter(torch.tensor(opacities_c, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity_duration_var = nn.Parameter(torch.tensor(opacities_v, dtype=torch.float, device="cuda").requires_grad_(True))
        
        # dynamic points
        self._xyz_motion = nn.Parameter(torch.tensor(motion_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation_motion = nn.Parameter(torch.tensor(motion_rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
        self.active_sh_degree = self.max_sh_degree
        
        # kinematic chain management
        plydata_shared = PlyData.read(path.replace('point_cloud.ply', 'shared_point_cloud.ply'))
        parent = np.asarray(plydata_shared.elements[0]["parent"], dtype=np.uint32).view(np.int32)
        level = np.asarray(plydata_shared.elements[0]["level"], dtype=np.uint32).view(np.int32)
        self._parent = torch.tensor(parent, dtype=torch.long, device="cuda").unsqueeze(-1)
        self._level = torch.tensor(level, dtype=torch.long, device="cuda").unsqueeze(-1)
        
        # Load duration data
        plydata_duration = PlyData.read(path.replace('point_cloud.ply', 'duration.ply'))
        duration = np.asarray(plydata_duration.elements[0]["duration"])[0]
        self.duration = duration

    def replace_tensor_to_optimizer(self, tensor_dict):
        optimizable_tensors = {}
        for name, tensor in tensor_dict.items():
            for group in self.optimizer.param_groups:
                if group["name"] == name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(tensor)
                        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                        del self.optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                        self.optimizer.state[group['params'][0]] = stored_state

                        optimizable_tensors[group["name"]] = group["params"][0]
                    else:
                        group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
                        optimizable_tensors[group["name"]] = group["params"][0]
                
        return optimizable_tensors

    def _prune_optimizer(self, static_mask, dynamic_mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].startswith("motion_"):
                mask = dynamic_mask
            else:
                mask = static_mask
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, static_mask, dynamic_mask=None):
        vaild_static_mask = ~static_mask
        if dynamic_mask is None:
            dynamic_mask = torch.zeros((self._xyz_motion.shape[0],) + static_mask.shape[1:], dtype=torch.bool, device=static_mask.device)
        
        # Splat that is a parent of other points should not be pruned
        n_static_points_prev = self._xyz.shape[0]
        splat_with_child = torch.unique(self._parent[(self._level > 0).squeeze() & torch.cat((~static_mask, ~dynamic_mask), dim=0)]) - n_static_points_prev
        assert torch.all(splat_with_child >= 0) # Only dynamic splats are parents
        
        parent_mask = torch.zeros_like(dynamic_mask)
        parent_mask[splat_with_child] = True
        dynamic_mask[splat_with_child] = False
        vaild_dynamic_mask = ~dynamic_mask
        
        parent_static = self._parent[:n_static_points_prev]
        parent_motion = self._parent[n_static_points_prev:]
        parent_static = parent_static[vaild_static_mask].clone()
        parent_motion = parent_motion[vaild_dynamic_mask].clone()
        
        level_static = self._level[:n_static_points_prev]
        level_motion = self._level[n_static_points_prev:]
        level_static = level_static[vaild_static_mask].clone()
        level_motion = level_motion[vaild_dynamic_mask].clone()
        
        # Mapper for updating old parent indices to new indices
        total_splats = self._parent.size(0)
        all_valid_mask = torch.cat([vaild_static_mask, vaild_dynamic_mask])
        mapper = torch.full((total_splats,), -1, dtype=torch.long, device=self._parent.device)
        mapper[all_valid_mask] = torch.arange(all_valid_mask.sum(), device=self._parent.device)
        
        # Update parent indices using the mapper
        new_parent = torch.full_like(parent_static, -1, dtype=torch.long)
        mask = parent_static != -1
        new_parent[mask] = mapper[parent_static[mask]]
        new_parent_motion = torch.full_like(parent_motion, -1, dtype=torch.long)
        mask = parent_motion != -1
        new_parent_motion[mask] = mapper[parent_motion[mask]]
        self._parent = torch.cat((new_parent, new_parent_motion), dim=0)
        self._level = torch.cat((level_static, level_motion), dim=0)
        
        optimizable_tensors = self._prune_optimizer(vaild_static_mask, vaild_dynamic_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        self._opacity_duration_center = optimizable_tensors["opacity_center"]
        self._opacity_duration_var = optimizable_tensors["opacity_var"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[vaild_static_mask]
        self.denom = self.denom[vaild_static_mask]
        self.xyz_error_accum = self.xyz_error_accum[vaild_static_mask]
        self.xyz_error_min = self.xyz_error_min[vaild_static_mask]
        self.xyz_error_min_timestamp = self.xyz_error_min_timestamp[vaild_static_mask]
        self.xyz_ssim_error_accum = self.xyz_ssim_error_accum[vaild_static_mask]
        self.error_denom = self.error_denom[vaild_static_mask]
        self.max_radii2D = self.max_radii2D[vaild_static_mask]
        self.min_radii2D = self.min_radii2D[vaild_static_mask]
        
        # dynamic points
        if vaild_dynamic_mask.shape[0] == 0:
            return
        
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # warning this may skip invaild paramgroups
            if not group["name"] in tensors_dict.keys():
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp,
                              new_opacity_duration_center, new_opacity_duration_var,
                              new_xyz_motion, new_rotation_motion,
                              ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,
        "opacity_center": new_opacity_duration_center,
        "opacity_var": new_opacity_duration_var,
        
        "motion_xyz" : new_xyz_motion,
        "motion_rotation" : new_rotation_motion,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        self._opacity_duration_center = optimizable_tensors["opacity_center"]
        self._opacity_duration_var = optimizable_tensors["opacity_var"]
        
        # dynamic points
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000

    def densification_postfix_onlystatic(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp,
                                         new_opacity_duration_center, new_opacity_duration_var,
                              ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacity,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "xyz_disp" : new_xyz_disp,
        "opacity_center": new_opacity_duration_center,
        "opacity_var": new_opacity_duration_var,
        }
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._xyz_disp = optimizable_tensors["xyz_disp"]
        self._opacity_duration_center = optimizable_tensors["opacity_center"]
        self._opacity_duration_var = optimizable_tensors["opacity_var"]
        
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.xyz_ssim_error_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.error_denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.min_radii2D = torch.ones((self._xyz.shape[0]), device="cuda") * 1000
    
    def densify_and_split(self, grads=None, grad_threshold=None, dynamic_grad_threshold=None, scene_extent=None, max_screen_size=None, max_dynamic_screen_size=None, N=2, target_static_mask=None, target_dynamic_mask=None):
        if target_static_mask is None:
            assert grads is not None
            assert grad_threshold is not None
            assert dynamic_grad_threshold is not None
            assert scene_extent is not None
        n_init_static_points = self._xyz.shape[0]
        
        if target_static_mask is None:
            # Extract points that satisfy the gradient condition
            padded_static_grad = torch.zeros((n_init_static_points), device="cuda")
            padded_static_grad[:grads.shape[0]] = grads.squeeze()
            
            selected_static_pts_mask = torch.where(padded_static_grad >= grad_threshold, True, False)
            childs = (self._level[:n_init_static_points] > 0).squeeze(-1)
            if torch.sum(childs) > 0:
                selected_static_pts_mask[childs] = torch.where(padded_static_grad[childs] >= dynamic_grad_threshold, True, False)
            selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                                torch.max(self.get_static_scaling, dim=1).values > self.percent_dense*scene_extent)
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * scene_extent
                selected_static_pts_mask = torch.logical_or(torch.logical_or(selected_static_pts_mask, big_points_vs), big_points_ws)
        else:
            selected_static_pts_mask = target_static_mask
        
        stds = self.get_static_scaling[selected_static_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_static_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_static_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_static_scaling[selected_static_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_static_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_static_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_static_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_static_pts_mask].repeat(N,1)
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask].repeat(N,1)
        new_opacity_duration_center = self._opacity_duration_center[selected_static_pts_mask].repeat(N,1,1)
        new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
        new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
        new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
        new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
        new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_static_pts_mask].repeat(N,1,1)) * 2
        
        new_xyz_error_min = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * 1000
        new_xyz_error_min_timestamp = torch.ones((selected_static_pts_mask.sum() * N, 1), device="cuda") * -1
        
        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])
            
        static_parent = self._parent[:n_init_static_points]
        static_level = self._level[:n_init_static_points]
        new_parent = self._parent[:n_init_static_points][selected_static_pts_mask].repeat(N,1)
        new_level = self._level[:n_init_static_points][selected_static_pts_mask].repeat(N,1)
        static_parent = torch.cat((static_parent, new_parent), dim=0)
        static_level = torch.cat((static_level, new_level), dim=0)
        
        dynamic_parent = self._parent[n_init_static_points:]
        dynamic_level = self._level[n_init_static_points:]
        
        self._parent = torch.cat((static_parent, dynamic_parent), dim=0)
        self._level = torch.cat((static_level, dynamic_level), dim=0)
        
        # Update parent index from additional static points
        dynamic_parent_mask = self._parent >= n_init_static_points
        self._parent[dynamic_parent_mask] += selected_static_pts_mask.sum() * N
        assert self._level.min() == 0
        
        self.densification_postfix_onlystatic(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp, new_opacity_duration_center, new_opacity_duration_var,
            )
        static_prune_filter = torch.cat((selected_static_pts_mask, torch.zeros(N * selected_static_pts_mask.sum(), device="cuda", dtype=bool)))
        dynamic_prune_filter = torch.zeros(self._xyz_motion.shape[0], device="cuda", dtype=bool)
        
        self.prune_points(static_prune_filter, dynamic_prune_filter)

    def densify_and_clone(self, grads, grad_threshold, dynamic_grad_threshold, scene_extent):
        # For the clone case, new splat is static child of the original splat
        n_init_static_points = self._xyz.shape[0]
        
        # Extract points that satisfy the gradient condition
        selected_static_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        childs = (self._level[:n_init_static_points] > 0).squeeze(-1)
        if torch.sum(childs) > 0:
            selected_static_pts_mask[childs] = torch.where(torch.norm(grads[childs], dim=-1) >= dynamic_grad_threshold, True, False)
        selected_static_pts_mask = torch.logical_and(selected_static_pts_mask,
                                            torch.max(self.get_static_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_static_pts_mask]
        new_features_dc = self._features_dc[selected_static_pts_mask]
        new_features_rest = self._features_rest[selected_static_pts_mask]
        new_opacity = self._opacity[selected_static_pts_mask]
        new_scaling = self._scaling[selected_static_pts_mask]
        new_rotation = self._rotation[selected_static_pts_mask]
        new_xyz_disp = self._xyz_disp[selected_static_pts_mask]
        new_opacity_duration_center = self._opacity_duration_center[selected_static_pts_mask]
        new_opacity_duration_center_len = ((new_opacity_duration_center[:, 1] - new_opacity_duration_center[:, 0]).abs() / 3).clamp_min(2/self.interval)
        new_opacity_duration_center[:, 1] = new_opacity_duration_center[:, 1] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 1])))
        new_opacity_duration_center[:, 0] = new_opacity_duration_center[:, 0] + (new_opacity_duration_center_len * (torch.randn_like(new_opacity_duration_center[:, 0])))
        new_opacity_duration_center = new_opacity_duration_center.clamp((self.time_shift + 1)/self.interval, (self.time_shift+self.duration - 1)/self.interval)
        new_opacity_duration_var = torch.ones_like(self._opacity_duration_var[selected_static_pts_mask]) * 2
        
        new_xyz_error_min = self.xyz_error_min[selected_static_pts_mask]
        new_xyz_error_min_timestamp = self.xyz_error_min_timestamp[selected_static_pts_mask]
        self.xyz_error_min = torch.cat([self.xyz_error_min, new_xyz_error_min])
        self.xyz_error_min_timestamp = torch.cat([self.xyz_error_min_timestamp, new_xyz_error_min_timestamp])
        
        # Inherit original splat's level and parent for static points
        new_parent_idx = torch.where(selected_static_pts_mask)[0]
        new_parent = self._parent[new_parent_idx]
        new_level = self._level[new_parent_idx]
        static_parent = self._parent[:n_init_static_points]
        static_level = self._level[:n_init_static_points]
        static_parent = torch.cat((static_parent, new_parent))
        static_level = torch.cat((static_level, new_level))
        
        dynamic_parent = self._parent[n_init_static_points:]
        dynamic_level = self._level[n_init_static_points:]
        
        self._parent = torch.cat((static_parent, dynamic_parent), dim=0)
        self._level = torch.cat((static_level, dynamic_level), dim=0)
        self._parent[self._parent >= n_init_static_points] += new_parent.shape[0]
        assert self._level.min() == 0
        
        self.densification_postfix_onlystatic(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz_disp, new_opacity_duration_center, new_opacity_duration_var,
        )
        
    def densify_and_prune(self, max_grad, max_dgrad, min_opacity, min_motion_opacity, extent,
                          max_screen_size, max_dynamic_screen_size, duration_thres=-5.0, 
                          s_max_ssim=0.5, s_l1_thres=0.1, d_max_ssim=0.5, d_l1_thres=0.1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, max_dgrad, extent)
        self.densify_and_split(grads, max_grad, max_dgrad, extent, max_screen_size, max_dynamic_screen_size)
        
        child_mask = (self._level > 0).squeeze()[:self._xyz.shape[0]]

        static_prune_mask = (self.get_opacity < min_opacity).squeeze()
        child_prune_mask = (self.get_opacity < min_motion_opacity).squeeze()
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * extent
            static_prune_mask = torch.logical_or(torch.logical_or(static_prune_mask, big_points_vs), big_points_ws)
        
        static_l1_mask = self.xyz_error_accum / self.error_denom.clamp(1e-4)
        static_l1_mask = static_l1_mask > s_l1_thres
        static_prune_mask = torch.logical_or(static_prune_mask, static_l1_mask.squeeze())
        
        static_ssim_mask = self.xyz_ssim_error_accum / self.error_denom.clamp(1e-4)
        static_ssim_mask = (static_ssim_mask < s_max_ssim) * (static_ssim_mask > 0)
        static_prune_mask = torch.logical_or(static_prune_mask, static_ssim_mask.squeeze())
        
        if max_dynamic_screen_size:
            big_points_vs = self.max_radii2D > max_dynamic_screen_size
            big_points_ws = self.get_static_scaling.max(dim=1).values > 0.1 * extent
            child_prune_mask = torch.logical_or(torch.logical_or(child_prune_mask, big_points_vs), big_points_ws)
        
        child_l1_mask = self.xyz_error_accum / self.error_denom.clamp(1e-4)
        child_l1_mask = child_l1_mask > d_l1_thres
        child_prune_mask = torch.logical_or(child_prune_mask, child_l1_mask.squeeze())
        
        child_ssim_mask = self.xyz_ssim_error_accum / self.error_denom.clamp(1e-4)
        child_ssim_mask = (child_ssim_mask < d_max_ssim) * (child_ssim_mask > 0)
        child_prune_mask = torch.logical_or(child_prune_mask, child_ssim_mask.squeeze())

        self.prune_points(static_prune_mask)

        torch.cuda.empty_cache()

    def prune_invisible(self):
        static_prune_mask = (self.xyz_error_min_timestamp < 0).squeeze()
        self.prune_points(static_prune_mask)

        torch.cuda.empty_cache()

    def prune_small(self):
        static_prune_mask = (self.min_radii2D < 5).squeeze()
        self.prune_points(static_prune_mask)

        torch.cuda.empty_cache()
        
    def add_densification_stats(self, viewspace_point_tensor, static_update_filter, static_num):
        self.xyz_gradient_accum[static_update_filter] += torch.norm(viewspace_point_tensor.grad[:static_num][static_update_filter,:2], dim=-1, keepdim=True)
        self.denom[static_update_filter] += 1
        
    def mark_prune_stats(self, radii, viewspace_point_error_tensor):
        static_num = self._xyz.shape[0]
        
        static_radii = radii[:static_num]
        static_vis_filter = viewspace_point_error_tensor.grad[:static_num, 0] > 0
        self.min_radii2D[static_vis_filter] = torch.min(self.min_radii2D[static_vis_filter], static_radii[static_vis_filter])
        
    def add_l1_ssim_stats(self, viewspace_point_error_tensor, static_update_filter, static_num, timestamp):
        static_errors = viewspace_point_error_tensor.grad[:static_num]
        static_l1_error = static_errors[static_update_filter, 1:2] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.xyz_error_accum[static_update_filter] += static_l1_error
        self.xyz_error_min_timestamp[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01), 
                                                                                 timestamp * torch.ones_like(static_l1_error), 
                                                                                 self.xyz_error_min_timestamp[static_update_filter])
        self.xyz_error_min[static_update_filter] = torch.where(torch.logical_and(self.xyz_error_min[static_update_filter] > static_l1_error, static_errors[static_update_filter, 0:1] > 0.01), 
                                                                       static_l1_error, self.xyz_error_min[static_update_filter])
        self.xyz_ssim_error_accum[static_update_filter] += static_errors[static_update_filter, 2:3] / static_errors[static_update_filter, 0:1].clamp_min(1e-4)
        self.error_denom[static_update_filter] += (static_errors[static_update_filter, 0:1] > 0).float()

    def build_connectivity_graph(self, timestamp, mask=None, disp_sim_threshold=None):
        def get_size_from_opacity(opacity, threshold):
            # Inverse compute gaussian's size from opacity. verify formula and check threshold with other function (prune?)
            size = torch.sqrt(2 * torch.log(opacity / threshold) / torch.pi)
            size[opacity < threshold] = 0.
            return size
        
        if mask is None:
            mask = torch.ones_like(self.get_opacity_at_t(0.0).squeeze(-1), dtype=torch.bool)

        # Compute pairwise distance between splats
        opacity_threshold = self.connectivity_threshold
        all_splats_xyz = self.get_xyz_at_t(timestamp)[mask].cpu()
        max_axis_rad3d = self.get_scaling()[mask].max(dim=-1)[0] # Maybe split long splats for this to work better?
        opacity_size_multiplier = get_size_from_opacity(self.get_opacity_at_t(timestamp)[mask].squeeze(-1), opacity_threshold)
        
        effective_size = (max_axis_rad3d * opacity_size_multiplier).cpu()
        disp = self._xyz_disp[mask[:self._xyz.shape[0]]]
        disp_np = disp.cpu().numpy()

        def build_overlap_graph(all_splats_xyz, effective_size):
            tree = cKDTree(all_splats_xyz.cpu())
            edges = []
            max_eff = effective_size.kthvalue(int(0.99 * effective_size.numel())).values.max()
            effective_size = effective_size.clamp_max(max_eff)

            for i, point in enumerate(all_splats_xyz):
                # Query candidates within a radius that surely covers possible overlaps.
                # Using effective_size[i] + max(effective_size) as an upper bound.
                candidate_idxs = tree.query_ball_point(point, effective_size[i] + max_eff)
                candidate_idxs = [j for j in candidate_idxs if j > i]
                candidates = all_splats_xyz[candidate_idxs]
                distances = np.linalg.norm(candidates - point, axis=1)
                valid_edges = distances < (effective_size[i].numpy() + effective_size[candidate_idxs].numpy())
                
                if disp_sim_threshold > 0:
                    disp_i = disp_np[i]
                    disp_candidates = disp_np[candidate_idxs]
                    # Avoid division by zero
                    disp_i_norm = np.linalg.norm(disp_i)
                    disp_candidates_norm = np.linalg.norm(disp_candidates, axis=1)
                    # If any norm is zero, set similarity to 0 (or handle as you wish)
                    nonzero = (disp_i_norm > 0) & (disp_candidates_norm > 0)
                    sim = np.zeros_like(disp_candidates_norm)
                    sim[nonzero] = np.dot(disp_candidates[nonzero], disp_i) / (disp_candidates_norm[nonzero] * disp_i_norm)
                    valid_disp = sim >= disp_sim_threshold

                    # Combine both filters
                    valid_edges = valid_edges & valid_disp

                edges.extend((i, j) for j, is_valid in zip(candidate_idxs, valid_edges) if is_valid)

            G = nx.Graph()
            G.add_edges_from(edges)
            all_nodes = set(range(len(all_splats_xyz)))
            G.add_nodes_from(all_nodes)
            return G
    
        G = build_overlap_graph(all_splats_xyz, effective_size)
        return G

    def create_dynamic_mask(self, viewpoint_loc, timestamp, vis_filter, extent, percentile=0.98, child_percentile=0.75, motion_thres=1000.0, min_motion_thres=1e-6, motion_thres_child=1000.0, min_motion_thres_child=1e-6):
        disp = self._xyz_disp[vis_filter].norm(dim=-1)
        disp_denorm = (self.get_xyz_at_t(timestamp, mode=0)[:self._xyz.shape[0]][vis_filter] - viewpoint_loc.to(self._xyz.device)).norm(dim=-1) ** 2
        disp = disp / (disp_denorm + 0.000001)
        disp = disp / (disp.max() + 0.000001)
        child_mask = (self._level[:self._xyz.shape[0]][vis_filter] > 0).squeeze()

        if child_percentile != 0.0:
            mv_thresh = torch.quantile(disp[~child_mask], percentile)
            child_mv_thresh = torch.quantile(disp[child_mask], child_percentile) if child_mask.sum() > 0 else 0
        else:
            mv_thresh = torch.quantile(disp, percentile)
            child_mv_thresh = mv_thresh if child_mask.sum() > 0 else 0

        child_dynamic_mask = child_mask & (disp > child_mv_thresh)

        unique_child_mask = torch.zeros_like(child_mask, dtype=torch.bool)
        for parent_id in torch.unique(self._parent[:self._xyz.shape[0]][vis_filter][child_mask]):
            child_indices = (self._parent[:self._xyz.shape[0]][vis_filter] == parent_id).nonzero(as_tuple=True)[0]
            if len(child_indices) == 1:
                unique_child_mask[child_indices] = True

        non_child_dynamic_mask = ~child_mask & (disp > mv_thresh)
        non_child_dynamic_mask = non_child_dynamic_mask | ((self._xyz_disp.norm(dim=-1)[vis_filter] > motion_thres * extent) & ~child_mask)
        non_child_dynamic_mask = non_child_dynamic_mask & (self._xyz_disp.norm(dim=-1)[vis_filter] > min_motion_thres * extent)
        
        child_dynamic_mask = child_dynamic_mask | ((self._xyz_disp.norm(dim=-1)[vis_filter] > motion_thres_child * extent) & child_mask)
        child_dynamic_mask = child_dynamic_mask & (self._xyz_disp.norm(dim=-1)[vis_filter] > min_motion_thres_child * extent)
        child_dynamic_mask = child_dynamic_mask & ~unique_child_mask
        dynamic_mask = child_dynamic_mask | non_child_dynamic_mask

        return dynamic_mask

    def extract_dynamic_points_from_static(self, viewpoint_loc, timestamp, vis_filter, extent, percentile=0.98, child_percentile=0.75, motion_thres=1000.0, min_motion_thres=1e-6, motion_thres_child=1000.0, min_motion_thres_child=1e-6,
                                           max_dur=None, disp_sim_threshold=0.0):
        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = max(float(max_dur), self.interval)
            
        to_dynamic_mask = self.create_dynamic_mask(viewpoint_loc, timestamp, vis_filter, extent, percentile, child_percentile, motion_thres, min_motion_thres, motion_thres_child, min_motion_thres_child)

        static_extract_mask = vis_filter.clone()
        static_extract_mask[vis_filter] = to_dynamic_mask
        static_extract_mask = torch.logical_and(static_extract_mask, self.xyz_error_min_timestamp.squeeze() >= 0)
        
        # Find unique parents of pruned splats and create families
        # Use elder's transform as the base transform for each family
        # NOTE: Index -1 is used to indicate that the splat is not a child of any other splat, split them based on proximity graph
        pruned_parent_ids = self._parent[:self._xyz.shape[0]][static_extract_mask].unique()
        families = {} 
        
        # Find mapping to first child in the family, which will be used as a base position for each family
        G = self.build_connectivity_graph(timestamp, static_extract_mask, disp_sim_threshold=disp_sim_threshold)
        
        # Create a mapper from indices in G to original indices
        node_to_splat_index_mapper = static_extract_mask.nonzero(as_tuple=True)[0]
        
        # Remove isolated nodes (nodes without any connected edges)
        # They will have index_to_component_mapper -1, and grouped together, handle them at the end
        isolated_nodes = [n for n in G.nodes if G.degree[n] == 0]
        G.remove_nodes_from(isolated_nodes)
        
        cc = nx.connected_components(G)
        
        for parent_id in pruned_parent_ids:
            child_indices = node_to_splat_index_mapper[(self._parent[:self._xyz.shape[0]][static_extract_mask] == parent_id).nonzero(as_tuple=True)[0]]
            families[parent_id.item()] = child_indices.tolist()
        
        # Split family if child belongs to different connected components
        index_to_component_mapper = torch.full((self._xyz.shape[0],), -1, dtype=torch.long, device=self._parent.device)
        for component_id, component in enumerate(cc):
            component_set = set(component)
            component_member_indices = node_to_splat_index_mapper[list(component_set)]
            index_to_component_mapper[component_member_indices] = component_id

        mapper_elder = torch.full((self._xyz.shape[0],), -1, dtype=torch.long, device=self._parent.device)
        for parent_id, child_indices in families.items():
            child_components = index_to_component_mapper[child_indices]
            valid_components = torch.unique(child_components)

            for component_id in valid_components:
                component_children = [idx for idx, comp_id in zip(child_indices, child_components) if comp_id == component_id]
                if len(component_children) >= 1:
                    # Split family by assigning new parent ID to children in this component
                    new_elder = component_children[0]
                    mapper_elder[component_children] = new_elder
        
        # Handle isolated nodes: assign the elder of children corresponding to isolated nodes to themselves
        if len(isolated_nodes) > 0:
            isolated_indices = node_to_splat_index_mapper[isolated_nodes]
            mapper_elder[isolated_indices] = isolated_indices
                
        copy_to_dynamic_mask = static_extract_mask.clone() & (mapper_elder == torch.arange(static_extract_mask.shape[0], device=self._parent.device))
            

        if self.keyframe_num == 0:
            self.keyframe_num = math.ceil((max_dur + self.time_shift*2 + 1) / self.interval) + 1 + 2

        # Initialize property of new parent virtual dynamic points
        xyz_motion_kp = []
        t_start = - self.time_shift
        for i in range(self.keyframe_num):
            xyz_motion_kp.append(self.get_xyz_at_t(t_start + self.interval * i)[:self._xyz.shape[0]][copy_to_dynamic_mask])
        new_xyz_motion = torch.stack(xyz_motion_kp, dim=1)
        
        rotation_motion_kp = []
        for i in range(self.keyframe_num):
            rotation_motion_kp.append(self.get_rotation_at_t(t_start + self.interval * i)[:self._xyz.shape[0]][copy_to_dynamic_mask])
        new_rotation_motion = torch.stack(rotation_motion_kp, dim=1)
        
        # Use the parent splat's motion instead of elder's motion
        # In case the parent is not root
        non_root_parent_mask = (self._level[:self._xyz.shape[0]] > 0).squeeze()[copy_to_dynamic_mask]
        non_root_parent_idx = (self._parent[:self._xyz.shape[0]].squeeze()[copy_to_dynamic_mask])[non_root_parent_mask]
        if non_root_parent_mask.any():
            new_xyz_motion[non_root_parent_mask] = self._xyz_motion[non_root_parent_idx - self._xyz.shape[0]]
            new_rotation_motion[non_root_parent_mask] = self._rotation_motion[non_root_parent_idx - self._xyz.shape[0]]
        
        if self.motion_degree > 1:
            new_xyz_motion = torch.cat([new_xyz_motion, torch.zeros_like(new_xyz_motion).repeat(1, 1, self.motion_degree-1)], dim=-1)

        new_opacity_motion = torch.full_like(self._opacity[copy_to_dynamic_mask], -torch.inf) # Virtual splats should not be rendered
        t = self.xyz_error_min_timestamp[copy_to_dynamic_mask]
        min_time = 0

        # Centers are middle point of the timestamp and the offset, and the max duration
        new_opacity_duration_center = torch.stack([
            torch.ones_like(new_opacity_motion) * (t * 1 / 2 + self.time_shift ) / self.interval, 
            torch.ones_like(new_opacity_motion) * ( (max_dur + t.clamp_min(min_time) * 1) / 2 + self.time_shift) / self.interval,
            ], dim=1).clamp((min_time+self.time_shift+1)/self.interval, (self.time_shift+max_dur-1)/self.interval)
        new_opacity_duration_var = torch.stack([
            torch.ones_like(new_opacity_motion) * (t + self.time_pad) , 
            torch.ones_like(new_opacity_motion) * (max_dur - t + self.time_pad) ,
            ], dim=1)
        
        # Child splats inherit parent's opacity temporal weight
        full_prune_mask = torch.cat((copy_to_dynamic_mask, torch.zeros((self._xyz_motion.size(0)), device="cuda", dtype=torch.bool)))
        child_splats = full_prune_mask & (self._level > 0).squeeze(-1)
        prune_child_mask = (self._level[:self._xyz.size(0)][copy_to_dynamic_mask] > 0).squeeze(-1)
        if torch.sum(child_splats) > 0:
            new_opacity_duration_center[prune_child_mask] = self._opacity_duration_center[self._parent[child_splats].squeeze(-1) - self._xyz.size(0)]
            new_opacity_duration_var[prune_child_mask] = self._opacity_duration_var[self._parent[child_splats].squeeze(-1) - self._xyz.size(0)]
        
        child_splats_mask = (self._level[:self._xyz.shape[0]] > 0).squeeze(-1)
        child_promotion_mask = child_splats_mask[copy_to_dynamic_mask]
        new_opacity_duration_center[child_promotion_mask] = self._opacity_duration_center[copy_to_dynamic_mask & child_splats_mask].detach()
        new_opacity_duration_var[child_promotion_mask] = self._opacity_duration_var[copy_to_dynamic_mask & child_splats_mask].detach()
        
        d = {
            "motion_xyz" : new_xyz_motion,
            "motion_rotation" : new_rotation_motion,
        }
        
        static_num = self._xyz.shape[0]

        new_parent_motion = -1 * torch.ones_like(self._parent[:static_num][copy_to_dynamic_mask])
        new_level_motion = torch.zeros_like(self._level[:static_num][copy_to_dynamic_mask])
        
        # Static parent is modified later
        static_parent = self._parent[:static_num]
        static_level = self._level[:static_num]
        
        dynamic_parent = self._parent[static_num:]
        dynamic_level = self._level[static_num:]
        
        # Parents are all dynamic points, indices do not change before pruning
        dynamic_parent = torch.cat((dynamic_parent, new_parent_motion))
        dynamic_level = torch.cat((dynamic_level, new_level_motion))
        
        self._parent = torch.cat((static_parent, dynamic_parent), dim=0)
        self._level = torch.cat((static_level, dynamic_level), dim=0)
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # dynamic points 
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        new_splat_indices = self._xyz.size(0) + self._xyz_motion.size(0) - copy_to_dynamic_mask.sum() + torch.arange(copy_to_dynamic_mask.sum(), device="cuda")
        
        self._opacity_duration_center[copy_to_dynamic_mask] = new_opacity_duration_center
        self._opacity_duration_var[copy_to_dynamic_mask] = new_opacity_duration_var
        
        extract_only_elder_mapper = mapper_elder[static_extract_mask]
            
        # Edit child splats' properties
        root_level_mask = (self._level[:self._xyz.size(0)] == 0).squeeze(-1)
        base_extract_only_elder_mapper = mapper_elder[static_extract_mask & root_level_mask]
        self._opacity_duration_center[static_extract_mask & root_level_mask] = self._opacity_duration_center[base_extract_only_elder_mapper]
        self._opacity_duration_var[static_extract_mask & root_level_mask] = self._opacity_duration_var[base_extract_only_elder_mapper]
            
        # Re-define coordinates based on elder splat
        root_child_mask = (self._level[:self._xyz.size(0)] == 0).squeeze(-1)
        root_child_mask_pruned = root_child_mask[static_extract_mask]
        inv_elder_rot = quaternion_invert(self.rotation_activation(self._rotation[extract_only_elder_mapper]))[root_child_mask_pruned]
        self._rotation[static_extract_mask & root_child_mask] = quaternion_multiply(inv_elder_rot, self.rotation_activation(self._rotation[static_extract_mask][root_child_mask_pruned]))
        self._xyz[static_extract_mask & root_child_mask] = quaternion_apply(inv_elder_rot, self._xyz[static_extract_mask][root_child_mask_pruned] - self._xyz[extract_only_elder_mapper][root_child_mask_pruned])

        self._parent[:self._xyz.size(0)][copy_to_dynamic_mask] = new_splat_indices.unsqueeze(-1)
        self._parent[:self._xyz.size(0)][static_extract_mask] = self._parent[:self._xyz.size(0)][extract_only_elder_mapper]
        
        self._level[:self._xyz.size(0)][static_extract_mask] = 1
        
    def prune_nan_points(self):
        if self._rotation_motion.shape[0] > 0 and \
            torch.isclose(self._rotation_motion, torch.zeros_like(self._rotation_motion)).all(dim=-1).any(dim=-1).any():
            print("Invalid rotation detected")
            from IPython import embed; embed()
        
        if self._xyz.shape[0] > 0 and self._xyz.isnan().any():
            static_prune_mask = self._xyz.isnan().any(dim=-1)
        else:
            static_prune_mask =  torch.zeros(self._xyz.size(0), device=self._xyz.device, dtype=torch.bool)
        if self._xyz_motion.shape[0] > 0 and self._xyz_motion.isnan().any():
            dynamic_prune_mask = self._xyz_motion.isnan().flatten(start_dim=1).any(dim=-1) | \
                self._rotation_motion.isnan().flatten(start_dim=1).any(dim=-1)
        else:
            dynamic_prune_mask =  torch.zeros(self._xyz_motion.size(0), device=self._xyz_motion.device, dtype=torch.bool)
        
        if static_prune_mask.sum() + dynamic_prune_mask.sum() > 0:
            self.prune_points(static_prune_mask, dynamic_prune_mask)


    def expand_duration(self, duration):
        duration = int(duration)+1
        if duration <= self.duration:
            return False
        
        # no dynamic points
        if self._xyz_motion.shape[0] == 0:
            if self.disp_consistency:
                d = {"xyz_disp": self._xyz_disp / self.duration * duration}
                optimizable_tensors = self.replace_tensor_to_optimizer(d)
                self._xyz_disp = optimizable_tensors["xyz_disp"]
            self.duration = duration
            return False
        
        require_dim = math.ceil((duration + self.time_shift + self.time_pad*2 + 1) / self.interval) + 1 + 2
        cur_dim = self._xyz_motion.shape[1]
        num_expand = require_dim - cur_dim
        
        if num_expand < 1:
            if self.disp_consistency:
                d = {"xyz_disp": self._xyz_disp / self.duration * duration}
                optimizable_tensors = self.replace_tensor_to_optimizer(d)
                self._xyz_disp = optimizable_tensors["xyz_disp"]
            self.duration = duration
            return False
            
        # linear linterpolation of lastest frame
        def lin_interp_last(x, n, zero_init=False, average=1):
            diff = (x[:, -average:] - x[:, -average-1:-average]).mean(dim=1, keepdim=True) * 1.0
            new_frames = torch.arange(1, n+1, device="cuda").view(1, -1, * [1] * len(diff.shape[2:])) * diff + x[:, -1:]
            if self.motion_degree > 1 and zero_init:
                new_frames[..., 3:] = 0.
            return torch.cat([x, new_frames], dim=1)

        num_avg = min(self.keyframe_num-2, 4)
        new_xyz_motion = lin_interp_last(self._xyz_motion, num_expand, zero_init=True, average=num_avg)
        new_rotation_motion = lin_interp_last(self._rotation_motion, num_expand, average=num_avg)
        
        ones_var = torch.ones_like(self._opacity_duration_var[:, 1])
        
        # Handle static child opacity
        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center + self.time_shift / self.interval > (duration + self.time_shift) / self.interval - 0.5).any(dim=1),
                                                   ones_var, self._opacity_duration_var[:, 1])
        new_opacity_duration_center = self._opacity_duration_center.clamp_max((self.time_shift+self.duration - 1)/self.interval)
        
        d = {
            "motion_xyz" : new_xyz_motion,
            "motion_rotation" : new_rotation_motion,
            "opacity_center" : new_opacity_duration_center,
            "opacity_var" : new_opacity_duration_var,
        }
        if self.disp_consistency:
            d["xyz_disp"] = self._xyz_disp / self.duration * duration
            
        optimizable_tensors = self.replace_tensor_to_optimizer(d)
        
        self._xyz_motion = optimizable_tensors["motion_xyz"]
        self._opacity_duration_center = optimizable_tensors["opacity_center"]
        self._opacity_duration_var = optimizable_tensors["opacity_var"]
        self._rotation_motion = optimizable_tensors["motion_rotation"]
        
        if self.disp_consistency:
            self._xyz_disp = optimizable_tensors["xyz_disp"]
            
        self.keyframe_num = require_dim
        self.duration = duration

        return True
    
    def mark_error(self, loss, timestamp):
        t_idx = timestamp // self.interval
        
        if t_idx in self.error_dict.keys():
            self.error_dict[t_idx] = (self.error_dict[t_idx][0] + loss, self.error_dict[t_idx][1] + 1)
        else:
            self.error_dict[t_idx] = (loss, 1)
    
    def get_errorneous_timestamp(self):
        if len(self.error_dict) == 0:
            return None
        
        max_loss = 0
        max_idx = 0
        max_count = 0
        
        for t_idx in self.error_dict.keys():
            loss, count = self.error_dict[t_idx]
            max_count = max(max_count, count)
            
            if loss / count > max_loss and count > max_count * 0.1:
                max_loss = loss / count
                max_idx = t_idx
        
        if max_loss == 0:
            return None
        
        del self.error_dict[max_idx]
        
        return (max_idx + 0.5) * self.interval
 
    def adjust_temp_opa(self, max_dur=None):
        if max_dur is None:
            max_dur = self.duration
        else:
            max_dur = float(max_dur)
        if self._xyz_motion.shape[0] == 0:
            return
        
        # Handle temporal opacity for static child splats
        new_opacity_duration_var = self._opacity_duration_var.detach().clone()
        new_opacity_duration_var[:, 1] = torch.where((self._opacity_duration_center > (max_dur + self.time_shift) / self.interval - 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 1].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 1])
        new_opacity_duration_var[:, 0] = torch.where((self._opacity_duration_center < (self.time_shift) / self.interval + 0.2).any(dim=1),
                                                   self._opacity_duration_var[:, 0].clamp_min(1)*2,
                                                   self._opacity_duration_var[:, 0])
        new_opacity_duration_center = self._opacity_duration_center.clamp((self.time_shift) / self.interval + 0.2, (max_dur + self.time_shift) / self.interval - 0.2)
        
        new_opacity_duration_var = torch.where(self._opacity_duration_var < torch.tensor(0.5),
                                                   torch.ones_like(self._opacity_duration_var)*torch.tensor(0.5),
                                                   new_opacity_duration_var)
        
        d = {
            "opacity_center" : new_opacity_duration_center,
            "opacity_var" : new_opacity_duration_var,
        }
            
        optimizable_tensors = self.replace_tensor_to_optimizer(d)
        
        self._opacity_duration_center = optimizable_tensors["opacity_center"]
        self._opacity_duration_var = optimizable_tensors["opacity_var"]