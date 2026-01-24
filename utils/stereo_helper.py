import sys 
import os
import torch
import numpy as np 
import warnings
import cv2

from utils.camera_utils import pix2ray, ray2pix
from utils.general_utils import inverse_sigmoid
from utils.stereo_utils import add_gaussians, xy_rotmat, ellipse_endpoints_parallel, get_error_areas
import torchvision
from scipy.spatial.transform import Rotation as R
import random
from utils.sh_utils import RGB2SH
from time import time
from torch.nn import functional as F
from scene import im_reader

warnings.filterwarnings("ignore")

# Get ellipse's scale at 3D space, using back projecting to z=1 plane
def get_ellipse_scale(valid_ellipses, height, width, project_inv):
    major1, major2, minor1, minor2 = ellipse_endpoints_parallel(valid_ellipses)
    major1, major2, minor1, minor2 = torch.from_numpy(major1).cuda().float(), torch.from_numpy(major2).cuda().float(), \
        torch.from_numpy(minor1).cuda().float(), torch.from_numpy(minor2).cuda().float()
    major1ray = pix2ray(major1[:, 0], major1[:, 1], height, width, project_inv)
    major2ray = pix2ray(major2[:, 0], major2[:, 1], height, width, project_inv)
    minor1ray = pix2ray(minor1[:, 0], minor1[:, 1], height, width, project_inv)
    minor2ray = pix2ray(minor2[:, 0], minor2[:, 1], height, width, project_inv)
    
    # Get X, Y at Z=1 plane, do we need to divide it with w?
    major1ray = major1ray[:, :3] / major1ray[:, 3:4]
    major1ray = major1ray[:, :2] / major1ray[:, 2:3]
    major2ray = major2ray[:, :3] / major2ray[:, 3:4]
    major2ray = major2ray[:, :2] / major2ray[:, 2:3]
    minor1ray = minor1ray[:, :3] / minor1ray[:, 3:4]
    minor1ray = minor1ray[:, :2] / minor1ray[:, 2:3]
    minor2ray = minor2ray[:, :3] / minor2ray[:, 3:4]
    minor2ray = minor2ray[:, :2] / minor2ray[:, 2:3]
    
    ellipse_xscale = torch.norm(major1ray - major2ray, dim=1) / 2.0
    ellipse_yscale = torch.norm(minor1ray - minor2ray, dim=1) / 2.0
    # ellipse_angle = torch.atan2(major2ray[:, 1] - major1ray[:, 1], major2ray[:, 0] - major1ray[:, 0])
    
    return ellipse_xscale, ellipse_yscale

def get_splat_props_for_depth(depth, directioninlocal, camera2world):
    # depth: [n] or [n, e]
    # directioninlocal: [e, 4]
    # camera2world: [4, 4]
    directioninlocal = directioninlocal.clone() / directioninlocal[:, -1:]
    targetPz = torch.tensor(depth).cuda() # [n]
    if len(targetPz.shape) == 1:
        targetPz = targetPz.unsqueeze(1)
    rate = targetPz / directioninlocal[:, 2].unsqueeze(0) # [n, e] 
    
    localpoint = directioninlocal.unsqueeze(0) * rate.unsqueeze(2) # [n, e, 4]
    localpoint[:, :, -1] = 1
    
    n = localpoint.shape[0]
    e = localpoint.shape[1]
    localpoint = localpoint.reshape(n * e, 4)
    worldpointH = localpoint @ camera2world.T  #myproduct4x4batch(localpoint, camera2world) # 
    worldpoint = worldpointH / worldpointH[:, 3:] #
    
    xyz = worldpoint[:, :3]
    distancetocameracenter = torch.norm(localpoint[:, :3], dim=1)
    
    log_distance = torch.log(distancetocameracenter)
    # log_distance = log_distance - 1.0 # why -1.0?
    
    xyz = xyz.reshape(n, e, 3)
    log_distance = log_distance.reshape(n, e)
    
    return xyz, log_distance

def get_depth_uv_visibility(test_depth, directioninlocal, camera2world, second_cam_w2c, second_cam_project_mat, height, width):
    xyz, log_distance = get_splat_props_for_depth(test_depth.reshape(1), directioninlocal, camera2world)
    worldpoint = torch.cat((xyz, torch.ones((xyz.shape[0], 1), device="cuda")), dim=1)
    second_cam_xyz = worldpoint @ second_cam_w2c.T
    second_u, second_v = ray2pix(second_cam_xyz, height, width, second_cam_project_mat)
    
    u_sign = 1 if second_u >= height else (-1 if second_u < 0 else 0)
    v_sign = 1 if second_v >= width else (-1 if second_v < 0 else 0)
    return u_sign, v_sign
            
def sample_cameras(mv_distance_threshold, num_samples, viewpointset):
    while True:
        num_cameras = len(viewpointset)
        cams = random.sample(range(num_cameras), num_samples)
        cam0 = viewpointset[cams[0]]
        cam1 = viewpointset[cams[1]]
        cam_distance = np.linalg.norm(cam0.T - cam1.T)
        if cam_distance > mv_distance_threshold:
            break
    return cam0, cam1

def extract_candidate_gaussians(cam0, cam1, gaussians, scene, pipe, dataset, background, render,
                            iteration=None,
                            check_depth=True, depth_kernel=1, margin_depth=1,
                            error_abs_thres=0.02, error_rel_thres=0.05, opacity_threshold=0.9, dynamic_leniency=-1.0, gaussian_blur_size=3):
    error_ellipses = []
    error_colors = []
    
    images = []
    gt_images = []
    pixel_depths = []
    
    if gaussian_blur_size % 2 == 0:
        gaussian_blur_size += 1
    
    for viewpoint_cam in [cam0, cam1]:
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, near=dataset.near, far=dataset.far)
        image = render_pkg["render"]
        path = viewpoint_cam.image_path
        gt_image =  (im_reader(path, viewpoint_cam.resolution, viewpoint_cam.im_scale)).cuda()
        
        # Depth is not distorted, use it as raw
        depth_image = render_pkg["depth"]
        
        if dynamic_leniency > 0.001:
            next_frame = (im_reader(
                scene.getNextFrame(viewpoint_cam, scale=viewpoint_cam.im_scale, split="train", step=2).image_path, 
                viewpoint_cam.resolution, 
                viewpoint_cam.im_scale
            )).cuda()
            
            prev_frame = (im_reader(
                scene.getPrevFrame(viewpoint_cam, scale=viewpoint_cam.im_scale, split="train", step=2).image_path, 
                viewpoint_cam.resolution, 
                viewpoint_cam.im_scale
            )).cuda()
            
            if gaussian_blur_size > 1:
                blurred_gt_image = torchvision.transforms.functional.gaussian_blur(gt_image.clone(), kernel_size=[gaussian_blur_size, gaussian_blur_size], sigma=[1.0, 1.0])
                blurred_prev_frame = torchvision.transforms.functional.gaussian_blur(prev_frame.clone(), kernel_size=[gaussian_blur_size, gaussian_blur_size], sigma=[1.0, 1.0])
                blurred_next_frame = torchvision.transforms.functional.gaussian_blur(next_frame.clone(), kernel_size=[gaussian_blur_size, gaussian_blur_size], sigma=[1.0, 1.0])
            else:
                blurred_gt_image = gt_image
                blurred_prev_frame = prev_frame
                blurred_next_frame = next_frame

            dynamic_mask = (torch.sum(torch.abs(blurred_gt_image - blurred_prev_frame), dim=0) > dynamic_leniency) | \
                           (torch.sum(torch.abs(blurred_gt_image - blurred_next_frame), dim=0) > dynamic_leniency)

        else:
            dynamic_mask = None
        
        ellipses, cluster_points, ellipse_colors = get_error_areas(
            image, gt_image, abs_thres=error_abs_thres, rel_thres=error_rel_thres, dynamic_mask=dynamic_mask,
            blur_size=gaussian_blur_size
        )
        
        error_ellipses.append(ellipses)
        error_colors.append(ellipse_colors)
        
        pixel_depths.append(depth_image)
        
        images.append(image)
        gt_images.append(gt_image)
        
    views_new_xyzs = []
    views_new_features_dc = []
    views_new_scaling = []
    views_new_rotation = []
    views_new_opacity = []
    views_split_coords = []
    
    # Perform stereo matching to find new splat candidate
    # - Project some points to guess the epipolar line
    for i, viewpoint_cam in enumerate([cam0]):
        height = viewpoint_cam.image_height
        width = viewpoint_cam.image_width
        camera2world = viewpoint_cam.world_view_transform.T.inverse()
        project_mat = viewpoint_cam.projection_matrix.T
        project_inv = project_mat.inverse()
        
        # Assume undistorted image, draw epipolar line by projecting two points to the other image
        second_cam = cam0 if i == 1 else cam1
        second_cam_w2c = second_cam.world_view_transform.T
        second_cam_project_mat = second_cam.projection_matrix.T
        
        valid_indices = [idx for idx, ellipse in enumerate(error_ellipses[i]) if ellipse is not None]
        valid_ellipses = [error_ellipses[i][idx] for idx in valid_indices]
        valid_colors = [error_colors[i][idx] for idx in valid_indices]
        
        u, v = torch.tensor([ellipse.center[0] for ellipse in valid_ellipses]), torch.tensor([ellipse.center[1] for ellipse in valid_ellipses])
        u, v = u.cuda(), v.cuda()
        
        directioninlocal = pix2ray(u, v, height, width, project_inv)
        
        if len(valid_ellipses) == 0:
            continue
        
        rgbs = torch.stack(valid_colors, dim=0).cuda()
        featuredc = RGB2SH(rgbs)
        
        # Back-project 4 points at the end of axes of the ellipse
        elipse_xscale, elipse_yscale = get_ellipse_scale(valid_ellipses, height, width, project_inv)
        elipse_scales = torch.stack((elipse_xscale, elipse_yscale, torch.ones_like(elipse_xscale)), dim=1)
        
        # Transform mats are in format
        # [[R 0]
        # [T 1]]
        # Cam coord to World Coordinate
        axes_rotation = camera2world[:3, :3].T.inverse().cpu().numpy()
        
        rotations = []
        for ellipse in valid_ellipses:
            # Test XY rotation
            ellipse_rotmat = np.eye(3)
            ellipse_rotmat[:2, :2] = xy_rotmat(ellipse.angle + 90)
            ellipsoid_rotation = axes_rotation @ ellipse_rotmat
            
            ellipsoid_rotation = R.from_matrix(ellipsoid_rotation).as_quat()
            ellipsoid_rotation = ellipsoid_rotation[[3, 0, 1, 2]]
            rotations.append(torch.tensor(ellipsoid_rotation).cuda().float())
        
        rotation = torch.stack(rotations, dim=0)

        ellipse_logscale = torch.log(elipse_scales)

        # check time
        starttime = time()
        depth_marg_rate = 1.05
        depth_kernel_size = depth_kernel * 2 - 1
        
        # Get the top k smallest depths for the target pixels u, v
        k = min(margin_depth, depth_kernel_size * depth_kernel_size)
        unfolded_depths = F.unfold(pixel_depths[i].unsqueeze(0), kernel_size=depth_kernel_size, padding=depth_kernel_size//2)
        unfolded_depths = unfolded_depths.view(depth_kernel_size * depth_kernel_size, height, width)
        depths_at_uv = unfolded_depths[:, u.long(), v.long()]
        test_depths = torch.topk(depths_at_uv, k, dim=0, largest=False)[0]
        current_depths = pixel_depths[i][0, u.long(), v.long()]
        current_depth_indices = (test_depths == current_depths.unsqueeze(0)).long().argmax(dim=0)
        
        is_matched = torch.zeros((len(valid_ellipses)), dtype=torch.bool).cuda()
        need_split = torch.zeros((len(valid_ellipses)), dtype=torch.bool).cuda()
        featuredcs = featuredc
        
        xyzs = torch.zeros((len(valid_ellipses), 3), device="cuda")
        scaling = torch.zeros((len(valid_ellipses), 3), device="cuda")
        opacity = torch.zeros((len(valid_ellipses), 1), device="cuda")
        depths = torch.zeros((len(valid_ellipses)), device="cuda")
        second_us = torch.zeros((len(valid_ellipses)), device="cuda")
        second_vs = torch.zeros((len(valid_ellipses)), device="cuda")
        
        split_coords = torch.zeros((len(valid_ellipses), 3), device="cuda")
        
        # Run multiple depth in parallel here
        batch_xyzs, batch_log_distances = get_splat_props_for_depth(test_depths, directioninlocal, camera2world)
        batch_log_distances = torch.stack((batch_log_distances, batch_log_distances, batch_log_distances), dim=2)
        worldpoints = torch.cat((batch_xyzs, torch.ones((batch_xyzs.shape[0], batch_xyzs.shape[1], 1), device="cuda")), dim=2)
        
        batch_second_cam_xyz = worldpoints.reshape(-1, 4) @ second_cam_w2c.T
        batch_second_cam_xyz = batch_second_cam_xyz.reshape(batch_xyzs.shape[0], batch_xyzs.shape[1], 4)
        batch_second_u, batch_second_v = ray2pix(batch_second_cam_xyz.reshape(-1, 4), height, width, second_cam_project_mat)
        batch_second_u = batch_second_u.reshape(batch_xyzs.shape[0], batch_xyzs.shape[1])
        batch_second_v = batch_second_v.reshape(batch_xyzs.shape[0], batch_xyzs.shape[1])
        
        def visibility_check(u, v, width, height):
            u_visible = torch.logical_and(u >= 0, u < height)
            v_visible = torch.logical_and(v >= 0, v < width)
            visible = torch.logical_and(u_visible, v_visible)
            return visible
        
        def visible_depth_check(xyz, u, v, visible, pixel_depth, depth_marg_rate):
            view_depth = pixel_depth[0][u[visible].long(), v[visible].long()]
            cam_distance = xyz[visible][:, 2]
            if xyz.shape[-1] == 4:
                cam_distance /= xyz[visible][:, -1]
            cam_depth_inrange = depth_marg_rate * view_depth >= cam_distance
            visible[visible.clone()] = torch.logical_and(visible[visible].clone(), cam_depth_inrange)
            return visible
        
        batch_second_visible = visibility_check(batch_second_u, batch_second_v, width, height)
        
        first_view_error = torch.abs(images[i][:, u.long(), v.long()] - gt_images[i][:, u.long(), v.long()])
        first_view_error = first_view_error.mean(dim=0)
        
        def process_candidate(cand_idx):
            # Get new splat information
            xyz, log_distance = batch_xyzs[:, cand_idx], batch_log_distances[:, cand_idx]
            
            cand_u, cand_v = u[cand_idx].long(), v[cand_idx].long()
            gt_cand_pixels = gt_images[i][:, cand_u, cand_v].unsqueeze(1)
            
            second_u, second_v = batch_second_u[:, cand_idx], batch_second_v[:, cand_idx]
            visibility_mask = batch_second_visible[:, cand_idx]
            
            # Use the visible_depth_check function to check visibility
            if check_depth:
                visibility_mask = visible_depth_check(batch_second_cam_xyz[:, cand_idx], second_u, second_v, visibility_mask.clone(), pixel_depths[1-i], depth_marg_rate)

            second_u_viz = second_u[visibility_mask]
            second_v_viz = second_v[visibility_mask]
            
            second_u_viz = second_u_viz.long()
            second_v_viz = second_v_viz.long()
            
            splat_color = rgbs[cand_idx]
            
            # Check if projected center has the same color for each ray
            center_color = gt_images[1-i][:, second_u_viz, second_v_viz]
            center_color = center_color.permute(1, 0)
            view2_max_opacity, _ = torch.max(torch.abs(center_color - splat_color[:3].unsqueeze(0)), dim=1)
            view2_max_opacity = torch.ones_like(view2_max_opacity) - view2_max_opacity
            
            if visibility_mask.sum() == 0:
                return False
            
            max_possible_opacity = torch.zeros(len(test_depths[:, cand_idx]), dtype=torch.float32).cuda()
            max_possible_opacity[visibility_mask] = view2_max_opacity
            
            elipse_zscale = torch.exp(torch.min(ellipse_logscale[cand_idx, 0], ellipse_logscale[cand_idx, 1]))
            ellipse_logscale[cand_idx, 2] = torch.log(elipse_zscale) - 10.0
        
            split_coords[cand_idx] = batch_xyzs[current_depth_indices[cand_idx], cand_idx]
            
            if not visibility_mask[current_depth_indices[cand_idx]]:
                return False
            
            if max_possible_opacity[current_depth_indices[cand_idx]] < opacity_threshold:
                need_split[cand_idx] = True
            
            # Check estimated current depth first, to see if it matches)
            if torch.max(max_possible_opacity) > opacity_threshold:
                max_opacity_idx = torch.argmax(max_possible_opacity)
            
                is_matched[cand_idx] = True
                xyzs[cand_idx] = xyz[max_opacity_idx]
                opacity[cand_idx] = max_possible_opacity[max_opacity_idx]
                depths[cand_idx] = test_depths[:, cand_idx][max_opacity_idx]
                scaling[cand_idx] = log_distance[max_opacity_idx] + ellipse_logscale[cand_idx]
                second_us[cand_idx] = second_u[max_opacity_idx]
                second_vs[cand_idx] = second_v[max_opacity_idx]
                    
                return True
            else:
                return False

        for cand_idx in range(len(valid_ellipses)):
            process_candidate(cand_idx)
        
        bound_filter_cnt  = 0
        opacity_filter_cnt = 0

        # Ensure is_matched is a boolean tensor
        is_matched = is_matched.bool()
        need_split = need_split.bool()
    
        if is_matched.sum() > 0:
            orig_new_opacity = opacity[is_matched]
            new_xyz = xyzs[is_matched, :]
            new_features_dc = featuredcs[is_matched]
            new_scaling = scaling[is_matched, :]
            new_rotation = rotation[is_matched]
            new_opacity = inverse_sigmoid(orig_new_opacity)
        else:
            new_xyz = None
            
        if new_xyz is not None and new_xyz.shape[0] > 0:
            views_new_xyzs.append(new_xyz)
            views_new_features_dc.append(new_features_dc)
            views_new_scaling.append(new_scaling)
            views_new_rotation.append(new_rotation)
            views_new_opacity.append(new_opacity)
        
        new_split_coords = split_coords[need_split, :]
        if new_split_coords.shape[0] > 0:
            views_split_coords.append(new_split_coords)
    
    if len(views_split_coords) == 0:
        views_split_coords = torch.empty((0, 3), dtype=torch.float32, device="cuda")
    else:
        views_split_coords = torch.cat(views_split_coords, dim=0)
    
    if len(views_new_xyzs) == 0:
        return views_split_coords
    
    views_new_xyzs = torch.cat(views_new_xyzs, dim=0)
    views_new_features_dc = torch.cat(views_new_features_dc, dim=0).unsqueeze(1)
    views_new_scaling = torch.cat(views_new_scaling, dim=0)
    views_new_rotation = torch.cat(views_new_rotation, dim=0)
    views_new_opacity = torch.cat(views_new_opacity, dim=0)
        
    return views_new_xyzs, views_new_features_dc, views_new_scaling, views_new_rotation, views_new_opacity, views_split_coords

def correct_gaussians(target_cameras, gaussians, scene, pipe, dataset, background, render, mv_distance_threshold,
        iteration=None,
        depth_kernel=1,
        margin_depth=1,
        error_abs_thres=0.02,
        error_rel_thres=0.05,
        opacity_threshold=0.9,
        dynamic_leniency=-1.0,
        gaussian_blur_size=3,
        nearest_threshold=0.01,
        use_opacity_est=False,
    ):
    
    new_gss = []
    
    viewpointset = target_cameras
    unique_timestamps = set(cam.timestamp for cam in viewpointset)
    
    # Sort timestamps to ignore first few and last few
    sorted_timestamps = sorted(unique_timestamps)
    ignore_count = min(5, len(sorted_timestamps) // 2 - 1)
    valid_timestamps = sorted_timestamps[ignore_count:-ignore_count]
    
    unique_colmap_id = set(cam.colmap_id for cam in viewpointset)
    num_cameras = len(unique_colmap_id)
    used_cams = set()
    used_pairs = set()
    cams = []
    
    timestamps = []
    
    while True:
        timestamp = random.choice(valid_timestamps)
        cameras_at_timestamp = [cam for cam in viewpointset if cam.timestamp == timestamp]
        if len(cameras_at_timestamp) >= 2:
            break

    while True:
        cam0, cam1 = sample_cameras(mv_distance_threshold, 2, cameras_at_timestamp)
        if cam0.colmap_id not in used_cams:
            used_cams.add(cam0.colmap_id)
            break
        
    new_gs = extract_candidate_gaussians(cam0, cam1, gaussians, scene, pipe, dataset, background, render,
                            iteration=iteration,
                            depth_kernel=depth_kernel,
                            margin_depth=margin_depth,
                            error_abs_thres=error_abs_thres,
                            error_rel_thres=error_rel_thres,
                            opacity_threshold=opacity_threshold,
                            dynamic_leniency=dynamic_leniency,
                            gaussian_blur_size=gaussian_blur_size
            )
    
    new_xyzs = None
    if new_gs is None:
        return None
    elif isinstance(new_gs, torch.Tensor):
        new_split_coords = new_gs
    else:
        new_xyzs, new_features_dc, new_scaling, new_rotation, new_opacity, new_split_coords = new_gs
        new_timestamps = torch.full((new_xyzs.shape[0],), timestamp, device=new_xyzs.device)
    
    num_new_split_gaussians = 0
    split_target_gaussians = gaussians.get_nearest_gaussian(new_split_coords, timestamp, opacity_thres=nearest_threshold)
    valid_indices = split_target_gaussians[split_target_gaussians != -1].unique()
    target_gaussian_mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool, device="cuda")
    target_gaussian_mask[valid_indices[valid_indices < gaussians._xyz.shape[0]]] = True
    target_dynamic_mask = torch.zeros(gaussians._xyz_motion.shape[0], dtype=torch.bool, device="cuda")
    target_dynamic_mask[valid_indices[valid_indices >= gaussians._xyz.shape[0]] - gaussians._xyz.shape[0]] = True
    gaussians.densify_and_split(target_static_mask=target_gaussian_mask, target_dynamic_mask=target_dynamic_mask)
    num_new_split_gaussians += valid_indices.shape[0] * 2

    if not use_opacity_est: # 0.1 Opacity initialization is more stable
        new_opacity = None
    num_new_backproject_gaussians = 0
    if new_xyzs is not None:
        num_new_backproject_gaussians = add_gaussians(new_xyzs, new_features_dc, new_scaling, new_rotation, gaussians, new_timestamps, opacity=new_opacity, nearest_threshold=nearest_threshold)
    
    num_new_gaussians = num_new_split_gaussians + num_new_backproject_gaussians
    
    return timestamps, num_new_gaussians, cams
