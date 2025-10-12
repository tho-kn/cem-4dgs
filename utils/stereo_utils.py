import torch
import numpy as np
from .general_utils import inverse_sigmoid
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import label
import cv2
from sklearn.cluster import DBSCAN, KMeans
from pytorch3d.transforms import quaternion_apply, quaternion_multiply, quaternion_invert
import torchvision
import os
import matplotlib.pyplot as plt


def get_cam_distances(views):
    cam_distances = []
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            cam_distances.append(np.linalg.norm(views[i].T - views[j].T))
    cam_distances.sort()
    return cam_distances


def add_gaussians(xyz, featuredc, scales, rotations, gaussians, timestamps, opacity=None, xyz_disp=None, opacity_duration_center=None, opacity_duration_var=None, remove_root_splats=False, nearest_threshold=0.01):
    selectnumpoints = xyz.shape[0] # torch.sum(selectedmask).item()
    new_xyz = xyz
    new_features_dc = featuredc
    new_features_rest = torch.zeros((selectnumpoints, gaussians._features_rest.shape[1], gaussians._features_rest.shape[2])).cuda()
    new_scaling = scales
    new_rotation = rotations
    new_timestamp = timestamps
    new_opacity_duration_center = opacity_duration_center
    new_opacity_duration_var = opacity_duration_var
    
    if xyz_disp is None:
        new_xyz_disp = torch.zeros_like(xyz)
    else:
        new_xyz_disp = xyz_disp
    
    if opacity is None:
        new_opacity = inverse_sigmoid(0.9 *torch.ones_like(xyz[:, 0:1]))
    else:
        new_opacity = opacity
        
    nearest_gaussian = torch.full_like(new_xyz[:, 0], -1, dtype=torch.long)
    
    # Set gaussian properties relative to parent
    unique_timestamps = torch.unique(new_timestamp)
    for timestamp in unique_timestamps:
        mask = new_timestamp == timestamp
        time_nearest_gaussians = gaussians.get_nearest_gaussian(new_xyz[mask], timestamp.item(), opacity_thres=nearest_threshold).squeeze(-1)
        nearest_gaussian[mask] = time_nearest_gaussians
    min_time = 0
    max_dur = gaussians.duration

    if new_opacity_duration_center is None:
        new_opacity_duration_center = torch.stack([
            torch.ones_like(new_opacity) * (new_timestamp.unsqueeze(1) * 1 / 2 + gaussians.time_shift) / gaussians.interval, 
            torch.ones_like(new_opacity) * ( (max_dur + new_timestamp.clamp_min(min_time).unsqueeze(1) * 1) / 2 + gaussians.time_shift) / gaussians.interval,
            ], dim=1).clamp((min_time+gaussians.time_shift+1)/gaussians.interval, (gaussians.time_shift+max_dur-1)/gaussians.interval)
    if new_opacity_duration_var is None:
        new_opacity_duration_var = torch.stack([
            torch.ones_like(new_opacity) * (new_timestamp.unsqueeze(1) + gaussians.time_pad) , 
            torch.ones_like(new_opacity) * (max_dur - new_timestamp.unsqueeze(1) + gaussians.time_pad) ,
            ], dim=1)
    
    all_parent = gaussians._parent
    static_parent = all_parent[:gaussians._xyz.shape[0]]
    dynamic_parent = all_parent[gaussians._xyz.shape[0]:]

    all_level = gaussians._level
    static_level = all_level[:gaussians._xyz.shape[0]]
    dynamic_level = all_level[gaussians._xyz.shape[0]:]
    
    new_rotation_relative = torch.zeros_like(new_rotation)
    new_xyz_relative = torch.zeros_like(new_xyz)
    new_xyz_disp_relative = new_xyz_disp

    for timestamp in unique_timestamps:
        mask = new_timestamp == timestamp
        parent_indices = static_parent[nearest_gaussian[mask]].squeeze(-1)
        parent_indices[nearest_gaussian[mask] == -1] = -1
        
        inv_parent_rot = quaternion_invert(gaussians.get_rotation_at_t(timestamp.item(), mode=3)[parent_indices])
        new_rotation_relative[mask] = quaternion_multiply(inv_parent_rot, new_rotation[mask])
        
        parent_xyz = gaussians.get_xyz_at_t(timestamp.item(), mode=3)[parent_indices]
        new_xyz_relative[mask] = quaternion_apply(inv_parent_rot, new_xyz[mask] - parent_xyz)
    
    parent_is_root = (static_parent[nearest_gaussian] == -1).squeeze(-1) | (nearest_gaussian == -1)
    new_rotation_relative[parent_is_root] = new_rotation[parent_is_root]
    new_xyz_relative[parent_is_root] = new_xyz[parent_is_root]
    new_xyz_disp_relative[parent_is_root] = new_xyz_disp[parent_is_root]
    
    if remove_root_splats:
        new_rotation_relative = new_rotation_relative[~parent_is_root]
        new_xyz_relative = new_xyz_relative[~parent_is_root]
        new_xyz_disp_relative = new_xyz_disp_relative[~parent_is_root]
        new_features_dc = new_features_dc[~parent_is_root]
        new_features_rest = new_features_rest[~parent_is_root]
        new_scaling = new_scaling[~parent_is_root]
        new_rotation = new_rotation[~parent_is_root]
        new_timestamp = new_timestamp[~parent_is_root]
        new_opacity = new_opacity[~parent_is_root]
        new_xyz_disp = new_xyz_disp[~parent_is_root]
        new_opacity_duration_center = new_opacity_duration_center[~parent_is_root]
        new_opacity_duration_var = new_opacity_duration_var[~parent_is_root]
        nearest_gaussian = nearest_gaussian[~parent_is_root]
        selectnumpoints = new_xyz_relative.shape[0]

    # Set level and parent for new gaussians
    n_init_static_points = gaussians._xyz.shape[0]
    static_parent = torch.cat((static_parent, static_parent[nearest_gaussian]), dim=0)
    gaussians._parent = torch.cat((static_parent, dynamic_parent), dim=0)
    gaussians._parent[gaussians._parent >= n_init_static_points] += selectnumpoints
    
    static_level = torch.cat((static_level, static_level[nearest_gaussian]), dim=0)
    gaussians._level = torch.cat((static_level, dynamic_level), dim=0)
    
    new_xyz_error_min = torch.ones((selectnumpoints, 1), device="cuda") * 1000
    new_xyz_error_min_timestamp = torch.ones((selectnumpoints, 1), device="cuda") * -1
    
    gaussians.xyz_error_min = torch.cat([gaussians.xyz_error_min, new_xyz_error_min])
    gaussians.xyz_error_min_timestamp = torch.cat([gaussians.xyz_error_min_timestamp, new_xyz_error_min_timestamp])
    
    gaussians.densification_postfix_onlystatic(new_xyz_relative, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation_relative, new_xyz_disp_relative, new_opacity_duration_center, new_opacity_duration_var)
    
    return selectnumpoints

def xy_rotmat(angle):
    angle_rad = np.deg2rad(angle)
    rotmat = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                       [-np.sin(angle_rad), np.cos(angle_rad)]])
    return rotmat

def get_ellipse_rotmat(ellipse):
    return xy_rotmat(ellipse.angle)

def ellipse_endpoints_parallel(ellipses):
    center_xs = np.array([ellipse.center[0] for ellipse in ellipses])
    center_ys = np.array([ellipse.center[1] for ellipse in ellipses])
    axis_majors = np.array([ellipse.size[0] for ellipse in ellipses])
    axis_minors = np.array([ellipse.size[1] for ellipse in ellipses])
    angles = np.array([ellipse.angle for ellipse in ellipses])
    
    # Convert angle from degrees to radians
    angles_rad = np.deg2rad(angles)
    
    # Calculate the endpoints of the major axis
    major_x1s = center_xs + np.cos(angles_rad) * axis_majors / 2
    major_y1s = center_ys + np.sin(angles_rad) * axis_majors / 2
    major_x2s = center_xs - np.cos(angles_rad) * axis_majors / 2
    major_y2s = center_ys - np.sin(angles_rad) * axis_majors / 2
    
    # Calculate the endpoints of the minor axis
    minor_x1s = center_xs - np.sin(angles_rad) * axis_minors / 2
    minor_y1s = center_ys + np.cos(angles_rad) * axis_minors / 2
    minor_x2s = center_xs + np.sin(angles_rad) * axis_minors / 2
    minor_y2s = center_ys - np.cos(angles_rad) * axis_minors / 2
    
    return np.stack((major_x1s, major_y1s), axis=1), np.stack((major_x2s, major_y2s), axis=1), \
        np.stack((minor_x1s, minor_y1s), axis=1), np.stack((minor_x2s, minor_y2s), axis=1)

def ellipse_endpoints(ellipse):
    (center_x, center_y), (axis_major, axis_minor), angle = ellipse.center, ellipse.size, ellipse.angle
    
    # Convert angle from degrees to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate the endpoints of the major axis
    major_x1 = center_x + np.cos(angle_rad) * axis_major / 2
    major_y1 = center_y + np.sin(angle_rad) * axis_major / 2
    major_x2 = center_x - np.cos(angle_rad) * axis_major / 2
    major_y2 = center_y - np.sin(angle_rad) * axis_major / 2
    
    # Calculate the endpoints of the minor axis
    minor_x1 = center_x - np.sin(angle_rad) * axis_minor / 2
    minor_y1 = center_y + np.cos(angle_rad) * axis_minor / 2
    minor_x2 = center_x + np.sin(angle_rad) * axis_minor / 2
    minor_y2 = center_y - np.cos(angle_rad) * axis_minor / 2
    
    return (major_x1, major_y1), (major_x2, major_y2), (minor_x1, minor_y1), (minor_x2, minor_y2)
    
def fit_ellipse(points_in_cluster, height, width):
    ellipse = None
    margin = 1
    if points_in_cluster.shape[0] >= 2:  # Minimum number of points required by cv2.fitEllipse
        points_in_cluster_r = points_in_cluster.copy()
        points_in_cluster_r[:, 1] += margin
        points_in_cluster_u = points_in_cluster.copy()
        points_in_cluster_u[:, 0] += margin
        points_in_cluster_ru = points_in_cluster + margin
        points_in_cluster_fit = np.concatenate((points_in_cluster, points_in_cluster_r, points_in_cluster_u, points_in_cluster_ru), axis=0)
        ellipse = cv2.minAreaRect(points_in_cluster_fit)
        
        # Ignore if the ellipse eliment contains nan
        has_no_nan = not np.isnan(ellipse[0][0]) and not np.isnan(ellipse[0][1]) and not np.isnan(ellipse[1][0]) and not np.isnan(ellipse[1][1]) and not np.isnan(ellipse[2])
        # Ignore if the center of the ellipse is out of the image
        is_in_image = 0 <= ellipse[0][0] < height and 0 <= ellipse[0][1] < width
        is_positive = ellipse[1][0] > 0 and ellipse[1][1] > 0
        # Ignore overly large ellipses, and zero size ellipse
        max_area = points_in_cluster.shape[0]
        max_area += max(0.2 * points_in_cluster.shape[0], max(ellipse[1][0], ellipse[1][1]))
        if not has_no_nan:
            return None
        if not is_in_image:
            return None
        if not is_positive:
            return None
        if max_area <= ellipse[1][0] * ellipse[1][1] * np.pi / 4:
            return None
        return cv2.RotatedRect((ellipse[0][0] - 0.5 * margin, ellipse[0][1] - 0.5 * margin), (ellipse[1][0], ellipse[1][1]), ellipse[2])
    else:
        return cv2.RotatedRect((int(points_in_cluster[0, 0]), int(points_in_cluster[0, 1])), (1, 1), 0)

# Recursion in the paper is simplified, some hyperparameters changed
def hybrid_cluster(cluster_points, gt_image, recursive_rgb=False):
    gt_np = gt_image.cpu().numpy()
    splitted_clusters = []
    ellipses = []
    
    rgb_min_samples = 4
    force_multiple_cluster = [False for _ in range(len(cluster_points))]
    
    from concurrent.futures import ThreadPoolExecutor

    def process_cluster(cluster_point, force_multi):
        local_splitted_clusters = []
        local_ellipses = []
        new_clusters = []

        if cluster_point.shape[0] <= rgb_min_samples:
            ellipse = fit_ellipse(cluster_point, gt_image.shape[1], gt_image.shape[2])
            if ellipse is not None:
                local_splitted_clusters.append(cluster_point)
                local_ellipses.append(ellipse)
            return local_splitted_clusters, local_ellipses, new_clusters

        cluster_colors = gt_np[:, cluster_point[:, 0], cluster_point[:, 1]] * 255 * 0.5
        cluster_colors = cluster_colors.swapaxes(0, 1)
        cluster_xy = cluster_point
        cluster_vector = np.concatenate((cluster_xy, cluster_colors), axis=1)

        if force_multi:
            clustering = KMeans(n_clusters=2).fit(cluster_xy)
        else:
            clustering = DBSCAN(eps=4.0, min_samples=rgb_min_samples).fit(cluster_vector)

        cluster_labels = clustering.labels_
        unique_labels = set(cluster_labels)

        for label in unique_labels:
            if label != -1:
                if force_multi and recursive_rgb:
                    # Call DBSCAN again with splitted cluster
                    new_clusters.append((cluster_point[cluster_labels == label], False))
                else:
                    ellipse = fit_ellipse(cluster_point[cluster_labels == label], gt_image.shape[1], gt_image.shape[2])
                    if ellipse is None:
                        new_clusters.append((cluster_point[cluster_labels == label], True))
                    else:
                        local_splitted_clusters.append(cluster_point[cluster_labels == label])
                        local_ellipses.append(ellipse)

        return local_splitted_clusters, local_ellipses, new_clusters

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while len(cluster_points) > 0:
            cluster_point = cluster_points.pop()
            force_multi = force_multiple_cluster.pop()
            futures.append(executor.submit(process_cluster, cluster_point, force_multi))

        while futures:
            future = futures.pop(0)
            local_splitted_clusters, local_ellipses, new_clusters = future.result()
            splitted_clusters.extend(local_splitted_clusters)
            ellipses.extend(local_ellipses)
            for new_cluster, force_multi in new_clusters:
                futures.append(executor.submit(process_cluster, new_cluster, force_multi))
    
    return splitted_clusters, ellipses
    

def cluster_error_pixels(error_mask):
    # Find indices of error pixels
    error_indices = torch.nonzero(error_mask).numpy()
    
    # Use DBSCAN to cluster error pixel indices, Dense clustering
    clustering = DBSCAN(eps=1.5, min_samples=8, n_jobs=16).fit(error_indices)
    cluster_labels = clustering.labels_
    
    # Find cluster centers
    unique_labels = set(cluster_labels)
    cluster_points = []
    for label in unique_labels:
        if label != -1:  # -1 is for noise points
            points_in_cluster = error_indices[cluster_labels == label]
            cluster_points.append(points_in_cluster)
    
    # Second run for sparse clusters
    remainder_clusters = error_indices[cluster_labels == -1]
    clustering = DBSCAN(eps=1.5, min_samples=2, n_jobs=16).fit(remainder_clusters)
    cluster_labels = clustering.labels_
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label != -1:
            points_in_cluster = remainder_clusters[cluster_labels == label]
            cluster_points.append(points_in_cluster)
    
    return cluster_points

def get_error_areas(image, gt_image, dynamic_mask=None,
                    abs_thres=0.02, rel_thres=0.05, blur_size=3):
    # Img: 0~1
    if blur_size > 1:
        image_blurred = torchvision.transforms.functional.gaussian_blur(image.clone(), kernel_size=[blur_size, blur_size], sigma=[1.0, 1.0])
        gt_image_blurred = torchvision.transforms.functional.gaussian_blur(gt_image.clone(), kernel_size=[blur_size, blur_size], sigma=[1.0, 1.0])
    else:
        image_blurred = image.clone()
        gt_image_blurred = gt_image.clone()
    diff = torch.abs(image_blurred - gt_image_blurred)
    diff = torch.mean(diff, dim=0) # h, w
    
    # Apply gaussian smoothing to the error map
    diff = diff.cpu().numpy()
    diff = torch.tensor(diff)
    
    # Calculate absolute diff and percentile
    abs_diff = torch.abs(diff)
    diff_sorted, _ = torch.sort(diff.reshape(-1))
    numpixels = diff.shape[0] * diff.shape[1]
    percentile = torch.searchsorted(diff_sorted, diff.reshape(-1)).reshape(diff.shape) / numpixels
    
    # Create colormaps
    cmap_abs = plt.get_cmap('viridis')
    cmap_percentile = plt.get_cmap('plasma')
    
    # Calculate masks
    if dynamic_mask is not None:
        dynamic_diff = diff[dynamic_mask]
        dynamic_threshold = torch.sort(dynamic_diff.flatten())[0][int(dynamic_diff.numel() * (1 - rel_thres))].item()
        rel_outmask = diff >= dynamic_threshold
        outmask = torch.logical_and(rel_outmask, dynamic_mask.to(rel_outmask.device))
    else:
        threshold = diff_sorted[int(numpixels * (1 - rel_thres))].item() if rel_thres > 0 else 0
        rel_outmask = diff >= threshold
        outmask = rel_outmask
    abs_outmask = abs_diff >= abs_thres
    view_mask = torch.logical_and(outmask.to(abs_diff.device), abs_outmask)
    
    cluster_points = cluster_error_pixels(view_mask)
    cluster_points, ellipses = hybrid_cluster(cluster_points, gt_image)
    
    ellipse_colors = []
    n_ellipses = len(ellipses)
    
    for cluster_point, ellipse in zip(cluster_points[:n_ellipses], ellipses[:n_ellipses]):
        if ellipse is None:
            color = gt_image[:, cluster_point[:, 0], cluster_point[:, 1]]
            color = torch.mean(color, dim=1)
            ellipse_colors.append(color)
        else:
            color = gt_image[:, int(ellipse.center[0]), int(ellipse.center[1])]
            ellipse_colors.append(color)
            
    return ellipses, cluster_points, ellipse_colors
