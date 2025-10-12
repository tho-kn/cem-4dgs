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
import enum

import scipy
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal


def intrinsic_matrix(fx, fy, cx, cy):
    """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.],
    ])


class ProjectionType(enum.Enum):
    """Camera projection type (standard perspective pinhole or fisheye model)."""
    PERSPECTIVE = 'perspective'
    FISHEYE = 'fisheye'


def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s, per=1)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0]-1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

import torch
# pixel to normalized device coordinate
def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0

def ndc2pix(ndc, S):
    return ((ndc + 1.0) * S - 1.0) / 2.0

# pixel to ray direction in 3D homogeneous coordinates
def pix2ray(u, v, height, width, projectinverse):
    ndcu, ndcv = pix2ndc(u, height), pix2ndc(v, width)
            
    ndcu = ndcu.unsqueeze(1)
    ndcv = ndcv.unsqueeze(1)

    # Use inverse projection to get 3D ray for the 2D point
    ndccamera = torch.cat((ndcv, ndcu, torch.ones_like(ndcu), torch.ones_like(ndcu)), 1) # N,4 ...
    
    localpointuv = ndccamera @ projectinverse.T
    
    # 3D homogeneous coordinates normalized
    diretioninlocal = localpointuv / localpointuv[:,3:] # ray direction in camera space

    return diretioninlocal

def ray2pix(ray, height, width, project):
    ray = ray / ray[:, 3:]
    ndccamera = ray @ project.T
    ndcv, ndcu = ndccamera[:, 0] / ndccamera[:, 2], ndccamera[:, 1] / ndccamera[:, 2]
    return ndc2pix(ndcu, height), ndc2pix(ndcv, width)
