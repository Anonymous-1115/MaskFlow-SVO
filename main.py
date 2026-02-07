"""
    MaskFlow-SVO
"""

import os
import random
import sys

import torch
from torch import nn
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')

import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)

import g2o
import numpy as np
from scipy.spatial.transform import Rotation as R

from eval.kitti_odometry import KittiEvalOdom
from datasets.kitti import KITTI
from datasets.kitti_tracking import KITTI_Tracking
from models.network_swin_unet import MaskedPoseNet
from utils import *


class Cam:
    T_wc: np.ndarray
    T_cw: np.ndarray

    def __init__(self, T, dtype='wc', id=-1):
        assert T.shape == (4, 4)
        if dtype == 'wc':
            self.set_from_Twc(T)
        elif dtype == 'cw':
            self.set_from_Tcw(T)
        else:
            raise NotImplementedError

        self.id = id

    def set_from_Twc(self, T_wc):
        self.T_wc = T_wc
        self.T_cw = np.linalg.inv(self.T_wc)

    def set_from_Tcw(self, Tcw):
        self.T_cw = Tcw
        self.T_wc = np.linalg.inv(self.T_cw)


class Point:
    xyz: np.ndarray

    def __init__(self, xyz, color=None, id=-1):
        assert xyz.shape == (3,)
        self.xyz = xyz
        self.color = color
        self.bad = False

        self.id = id

    def set_xyz(self, xyz):
        self.xyz = xyz

    def set_color(self, color):
        self.color = color

    def is_bad(self):
        return self.bad

    def set_bad(self):
        self.bad = True


class Obs:
    obs_dtype = np.dtype({
        'names': ['uv', 'sigma2'],
        'formats': ['(2, )f4', 'f4']
    })

    matrix: np.ndarray

    def __init__(self, N_select, window_size):
        self.N_select = N_select
        self.window_size = window_size
        self.matrix = self.init_matrix(shape=(self.N_select, 1))

    @staticmethod
    def init_matrix(shape: tuple) -> np.ndarray:
        obs = np.empty(shape=shape, dtype=Obs.obs_dtype)
        obs[:, :] = np.nan
        return obs

    def expand_matrix(self) -> None:
        """
            we turn to a more efficient way, i.e. only maintain the sliding window
            so that the speed will not become slower and slower
        """
        h, w = self.matrix.shape
        assert h // self.N_select == w

        new_row = np.empty(shape=(self.N_select, w), dtype=Obs.obs_dtype)
        new_row[:, :] = np.nan
        new_matrix = np.vstack((self.matrix, new_row))  # (h + N_select, w)
        new_col = np.empty(shape=(h + self.N_select, 1), dtype=Obs.obs_dtype)
        new_col[:, :] = np.nan
        new_matrix = np.hstack((new_matrix, new_col))  # (h + N_select, w + 1)
        self.matrix = new_matrix

        h_new, w_new = self.matrix.shape
        if w_new > self.window_size:
            self.matrix = self.matrix[self.N_select:, 1:]

    def is_valid(self, i, j) -> bool:
        return ~np.isnan(self.matrix[i, j]['sigma2'])

    def set_uv(self, i, j, target: np.ndarray) -> None:
        assert target.shape == (2,)
        assert target.dtype == 'float32' or np.isnan(target).all()
        self.matrix[i, j]['uv'] = target

    def set_sigma(self, i, j, target: float) -> None:
        self.matrix[i, j]['sigma2'] = target


class WinMatch:
    """
        save historical matches in sliding window
        use in forward and backward matching
        each match is a python dictionary, with keys: 'flow' and 'mask'
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.forward_matches = []  # [t-r-1, t-1] -> t
        self.backward_matches = []  # t -> [t-r-1, t-1]

    def save_forward(self, forward_match):
        self.forward_matches.append(forward_match)

        # drop the oldest keyframe
        if len(self.forward_matches) >= self.window_size:
            self.forward_matches = self.forward_matches[1:]

    def save_backward(self, backward_match):
        self.backward_matches.append(backward_match)

        # drop the oldest keyframe
        if len(self.backward_matches) >= self.window_size:
            self.backward_matches = self.backward_matches[1:]


class WinRelativePose:
    """
        save historical VO relative poses in sliding window
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.T_12 = []
        self.T_21 = []
        self.sigma2 = []

    def save_sigma2(self, sigma2):
        self.sigma2.append(sigma2)
        if len(self.sigma2) >= self.window_size:
            self.sigma2 = self.sigma2[1:]

    def save_from_T12(self, T12):
        self.T_12.append(T12)
        T21 = np.linalg.inv(T12)
        self.T_21.append(T21)
        self.drop_if_necessary()

    def save_from_T21(self, T21):
        self.T_21.append(T21)
        T12 = np.linalg.inv(T21)
        self.T_12.append(T12)
        self.drop_if_necessary()

    def drop_if_necessary(self):
        # drop the oldest keyframe
        assert len(self.T_12) == len(self.T_21)
        if len(self.T_12) >= self.window_size:
            self.T_12 = self.T_12[1:]
            self.T_21 = self.T_21[1:]


class WinKeyFrame:
    """
        save images, disparities and depths in sliding window
    """

    def __init__(self, window_size):
        self.window_size = window_size
        self.images = []
        self.disps = []  # list of (1, H, W)
        self.depths = []

    def save_images(self, image):
        self.images.append(image)

        # drop the oldest keyframe
        if len(self.images) > self.window_size:
            self.images = self.images[1:]

    def save_disps(self, disp):
        self.disps.append(disp)

        # drop the oldest keyframe
        if len(self.disps) > self.window_size:
            self.disps = self.disps[1:]

    def save_depths(self, depth):
        self.depths.append(depth)

        # drop the oldest keyframe
        if len(self.depths) > self.window_size:
            self.depths = self.depths[1:]


class Dilate:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.maxpool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size - 1) // 2)

    def __call__(self, mask):
        """
            mask: (B, 1, H, W), 0 for mask
        """
        return -self.maxpool(-mask)


class Point2DFilter:
    def __init__(self, raw_size: tuple, grid_size: tuple):
        self.h, self.w = raw_size
        self.grid_h, self.grid_w = grid_size
        self.N_h = self.h // self.grid_h  # last row may have more
        self.N_w = self.w // self.grid_w  # last column may have more
        self.grids = [[[] for _ in range(self.N_w)] for _ in range(self.N_h)]

    def __call__(self, select_uv):
        """
            return: select idx
        """
        self.select_uv = select_uv
        self.idx = np.arange(len(select_uv))

        for i in self.idx:
            u = self.select_uv[i, 0]
            v = self.select_uv[i, 1]

            x = min(math.ceil(v / self.grid_h), self.N_h) - 1
            y = min(math.ceil(u / self.grid_w), self.N_w) - 1

            self.grids[x][y].append(i)

        results = []
        for i in range(self.N_h):
            for j in range(self.N_w):
                if len(self.grids[i][j]) == 0:
                    continue
                elif len(self.grids[i][j]) == 1:
                    results.append(self.grids[i][j][0])
                else:
                    results.append(random.choice(self.grids[i][j]))

        results.sort()

        return results


class SLAM:
    uv: torch.Tensor
    avg_flow_norm: float

    forward_flow: torch.Tensor
    M_flow_total_forward: torch.Tensor
    M_depth_total_forward: torch.Tensor
    M_neural_total_forward: torch.Tensor
    M_total_forward: torch.Tensor

    backward_flow: torch.Tensor
    M_flow_total_backward: torch.Tensor
    M_depth_total_backward: torch.Tensor
    M_neural_total_backward: torch.Tensor
    M_total_backward: torch.Tensor

    def __init__(self, args):
        self.args = args

        cfg = {
            'dataset': Config(path=args.dataset),
            'model': Config(path=args.model)
        }

        # for reproducibility and distribute training
        set_seed(args.seed)

        self.device = 'cuda'
        self.height = cfg['model'].network.height
        self.width = cfg['model'].network.width

        if args.seq is not None:
            cfg['dataset'].prediction.seq[0] = args.seq
        self.seq = cfg['dataset'].prediction.seq[0]

        # dataset and model
        self.dataset_name = cfg['dataset'].name
        if self.dataset_name == 'KITTI':
            self.dataset = KITTI(cfg, split='predict')
        elif self.dataset_name == 'KITTI_Tracking':
            self.dataset = KITTI_Tracking(cfg, split='predict', part='testing')
        else:
            raise NotImplementedError
        self.network = MaskedPoseNet(cfg)
        self.network.to(self.device).eval()

        ckpt = cfg['model'].network.ckpt
        if ckpt is not None:
            checkpoint = torch.load(ckpt, weights_only=False)
            self.network.load_state_dict(checkpoint['network'], strict=True)
            print('load checkpoint: {}'.format(ckpt))

        # matching network in backend
        self.cache_M_neural_forward = []
        self.cache_M_neural_backward = []

        # TODO: important parameters
        self.window_size = cfg['dataset'].backend.window_size  # 5
        self.N_select = cfg['dataset'].backend.N_select
        self.th_flow = cfg['dataset'].backend.th_flow
        self.th_depth = cfg['dataset'].backend.th_depth
        self.th_kf = cfg['dataset'].backend.th_kf
        self.th_nonzero_mask = cfg['dataset'].backend.th_nonzero_mask
        self.init_sigma = cfg['dataset'].backend.init_sigma
        self.sigma2 = cfg['dataset'].backend.sigma2
        self.th_score = cfg['dataset'].backend.th_score
        self.sigma2_scale = cfg['dataset'].backend.sigma2_scale
        self.sigma2_odom_rot = cfg['dataset'].backend.sigma2_odom_rot
        self.sigma2_odom_trans = cfg['dataset'].backend.sigma2_odom_trans
        self.sigma2_odom = cfg['dataset'].backend.sigma2_odom
        self.robust_kernel = cfg['dataset'].backend.robust_kernel
        self.use_uv_filter = cfg['dataset'].backend.use_uv_filter
        self.enable_timing = cfg['dataset'].backend.enable_timing
        self.num_overlap = cfg['dataset'].backend.num_overlap
        assert self.num_overlap < self.window_size

        # TODO: key buffers for the SLAM system
        self.init()
        self.keyframes = [0]  # frame id
        self.cameras = [Cam(np.eye(4), id=0)]
        self.points = [Point(xyz=nan(3), id=i) for i in range(self.N_select)]
        self.obs = Obs(self.N_select, self.window_size)
        self.window_matches = WinMatch(self.window_size)
        self.window_relative_poses = WinRelativePose(self.window_size)
        self.T_12 = np.eye(4)  # accumulated T_12 between two consecutive keyframes
        self.T_12_net = np.eye(4)
        self.T_wc = np.eye(4)  # current pose
        self.K = np.eye(3)
        self.bf = 0.0
        self.cur_disp = torch.zeros((1, 1, self.height, self.width), device=self.device)  # current disparity
        self.cnt_to_last_ba = 0

        # utils
        self.dilator = Dilate(kernel_size=5)
        self.uv_filter = Point2DFilter(raw_size=(self.height, self.width),
                                       grid_size=(5, 5))
        self.window_keyframes = WinKeyFrame(self.window_size)
        self.timestamps = [self.dataset[0].frame1.timestamp]
        self.traj_frontend = [np.eye(4)]
        self.traj_gt = [np.eye(4)]
        if hasattr(self.dataset, 'raw_timestamps'):
            self.raw_timestamps = self.dataset.raw_timestamps
            self.traj_raw_gt = self.dataset.raw_poses_all_seq[self.seq]
        self.residuals = [0. for _ in range(self.window_size - 1)]
        self.eval_tool = KittiEvalOdom()
        self.history_forward_masks = []

    def init(self):
        self.uv = init_uv(self.height, self.width, self.device)
        self.avg_flow_norm = 0.0
        self.T_12 = np.eye(4)
        self.T_12_net = np.eye(4)

        # forward
        self.forward_flow = torch.zeros((1, 2, self.height, self.width), device=self.device)
        self.M_flow_total_forward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_depth_total_forward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_neural_total_forward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_epi_total_forward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_total_forward = torch.ones((1, 1, self.height, self.width), device=self.device)

        # backward
        self.backward_flow = torch.zeros((1, 2, self.height, self.width), device=self.device)
        self.M_flow_total_backward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_depth_total_backward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_neural_total_backward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_epi_total_backward = torch.ones((1, 1, self.height, self.width), device=self.device)
        self.M_total_backward = torch.ones((1, 1, self.height, self.width), device=self.device)

    def clear_cache(self):
        self.cache_M_neural_forward = []
        self.cache_M_neural_backward = []

    def compute_forward_corr(self, flow_l1l2, flow_l2l1, depth1, M_neural_forward, prior_masks=None):
        uv1 = self.uv + self.forward_flow  # (B, 2, H, W), float
        M_flow_forward = flow_consistency_mask(uv1, flow_l1l2, flow_l2l1, th=self.th_flow)
        self.M_flow_total_forward = self.M_flow_total_forward * self.dilator(M_flow_forward)

        if prior_masks is not None:
            flow_mask = self.M_flow_total_forward.bool().logical_not()  # 取反
            act = (torch.logical_and(flow_mask, prior_masks)).any(dim=(-2, -1))[:, :, None, None]
            act_prior_mask = torch.logical_and(act, prior_masks).any(dim=1, keepdim=True)
            mask = torch.logical_or(flow_mask, act_prior_mask)
            self.M_flow_total_forward = mask.logical_not().float()  # 取反

        M_depth_forward = depth_mask(interpolate(uv1, depth1), d_min=self.th_depth[0], d_max=self.th_depth[1])
        self.M_depth_total_forward = self.M_depth_total_forward * M_depth_forward

        M_neural_forward = interpolate(uv1, M_neural_forward, mode='nearest')
        self.M_neural_total_forward = self.M_neural_total_forward * self.dilator(M_neural_forward)

        self.forward_flow = self.forward_flow + interpolate(uv1, flow_l1l2)
        self.M_total_forward = self.M_flow_total_forward * self.M_depth_total_forward * self.M_neural_total_forward * self.M_epi_total_forward

    def compute_backward_corr(self, flow_l2l1, flow_l1l2, depth2, M_neural_backward, prior_masks=None):
        M_flow_backward = flow_consistency_mask(self.uv, flow_l2l1, flow_l1l2, th=self.th_flow)
        self.M_flow_total_backward = interpolate(self.uv + flow_l2l1, self.M_flow_total_backward, mode='nearest') * self.dilator(M_flow_backward)

        if prior_masks is not None:
            flow_mask = self.M_flow_total_backward.bool().logical_not()  # 取反
            act = (torch.logical_and(flow_mask, prior_masks)).any(dim=(-2, -1))[:, :, None, None]
            act_prior_mask = torch.logical_and(act, prior_masks).any(dim=1, keepdim=True)
            mask = torch.logical_or(flow_mask, act_prior_mask)
            self.M_flow_total_backward = mask.logical_not().float()  # 取反

        M_depth_backward = depth_mask(depth2, d_min=self.th_depth[0], d_max=self.th_depth[1])
        self.M_depth_total_backward = interpolate(self.uv + flow_l2l1, self.M_depth_total_backward,
                                                  mode='nearest') * M_depth_backward

        self.M_neural_total_backward = interpolate(self.uv + flow_l2l1, self.M_neural_total_backward,
                                                   mode='nearest') * self.dilator(M_neural_backward)

        self.backward_flow = flow_l2l1 + interpolate(self.uv + flow_l2l1, self.backward_flow)
        self.M_total_backward = self.M_flow_total_backward * self.M_depth_total_backward * self.M_neural_total_backward * self.M_epi_total_backward

    def is_keyframe(self, i):
        avg_flow_norm_forward = avg_norm(self.forward_flow, mask=self.M_total_forward)
        avg_flow_norm_backward = avg_norm(self.backward_flow, mask=self.M_total_backward)
        self.avg_flow_norm = 0.5 * (avg_flow_norm_forward + avg_flow_norm_backward).item()

        # non-mask ratio
        ratio_forward = self.M_total_forward.sum() / (self.height * self.width)
        ratio_backward = self.M_total_backward.sum() / (self.height * self.width)
        ratio = max(ratio_forward, ratio_backward)

        if np.isnan(self.avg_flow_norm) or self.avg_flow_norm > self.th_kf or ratio < self.th_nonzero_mask:
            forward_match = {
                'flow': self.forward_flow,
                'mask': self.M_total_forward
            }
            backward_match = {
                'flow': self.backward_flow,
                'mask': self.M_total_backward
            }
            self.window_matches.save_forward(forward_match)
            self.window_matches.save_backward(backward_match)

            self.window_relative_poses.save_from_T12(self.T_12_net)

            self.init()
            self.keyframes.append(i)

            if len(self.keyframes) > self.window_size:
                self.cnt_to_last_ba = self.cnt_to_last_ba + 1

            return True

        return False

    def expand_graph(self, cur_pose, cur_depth) -> bool:
        # add camera
        cur_cam_id = len(self.cameras)
        self.cameras.append(Cam(cur_pose, id=cur_cam_id))  # new keyframe camera
        cur_cam_id = len(self.cameras) - 1
        cam_start_id = max(cur_cam_id - self.window_size + 1, 0)
        cur_cam_window_id = cur_cam_id - cam_start_id
        point_start_id = max(cam_start_id * self.N_select, 0)

        # add observations and 3D points
        self.obs.expand_matrix()
        is_first_pair = len(self.keyframes) == 2

        # (step 1) forward matching
        M_exist = torch.ones((1, 1, self.height, self.width), device=self.device)
        if not is_first_pair:
            for j in range(-1, -self.window_size, -1):
                host_cam_id = cur_cam_id + j
                if host_cam_id < 0:
                    break

                for k in range(self.N_select):
                    point_id = k + host_cam_id * self.N_select
                    point_window_id = point_id - point_start_id
                    if not self.obs.is_valid(point_window_id, cur_cam_window_id - 1):
                        continue

                    last_uv = self.obs.matrix[point_window_id, cur_cam_window_id - 1]['uv']
                    last_uv_ = torch.from_numpy(last_uv)[None, :, None, None].to(self.device)
                    forward_match = self.window_matches.forward_matches[-1]
                    can_match_forward = interpolate(last_uv_, forward_match['mask'],
                                                    mode='nearest').squeeze().bool().item()
                    if can_match_forward:
                        sigma2 = self.obs.matrix[point_window_id, cur_cam_window_id - 1]['sigma2']

                        backward_match = self.window_matches.backward_matches[-1]
                        flow_mask = flow_consistency_mask(self.uv, forward_match['flow'], backward_match['flow'],
                                                          th=self.th_flow)
                        if not interpolate(last_uv_, flow_mask, mode='nearest').squeeze().bool().item():
                            continue

                        forward_flow = interpolate(last_uv_, forward_match['flow'])
                        cur_uv_ = last_uv_ + forward_flow  # 2D position on cur

                        if not interpolate(cur_uv_, backward_match['mask'], mode='nearest').squeeze().bool().item():
                            continue

                        cur_uv = cur_uv_.squeeze().cpu().numpy()
                        if not is_in_image(cur_uv, self.height, self.width):
                            continue

                        # epipolar check
                        T_cur_last = self.window_relative_poses.T_21[cur_cam_window_id - 1]
                        s = epipolar_score(last_uv, cur_uv, T_cur_last, self.K, scale=1e3)
                        if s > self.th_score:
                            continue

                        self.obs.set_uv(point_window_id, cur_cam_window_id, cur_uv)
                        self.obs.set_sigma(point_window_id, cur_cam_window_id, sigma2 + self.sigma2)

                        M_exist[:, :, bound(math.floor(cur_uv[1]), 0, self.height - 1),
                        bound(math.floor(cur_uv[0]), 0, self.width - 1)] = 0
                        M_exist[:, :, bound(math.floor(cur_uv[1]), 0, self.height - 1),
                        bound(math.ceil(cur_uv[0]), 0, self.width - 1)] = 0
                        M_exist[:, :, bound(math.ceil(cur_uv[1]), 0, self.height - 1),
                        bound(math.floor(cur_uv[0]), 0, self.width - 1)] = 0
                        M_exist[:, :, bound(math.ceil(cur_uv[1]), 0, self.height - 1),
                        bound(math.ceil(cur_uv[0]), 0, self.width - 1)] = 0

            M_exist = self.dilator(M_exist)

        # (step 2) mapping
        M_select = self.window_matches.backward_matches[-1]['mask'] * M_exist
        if M_select.sum().item() == 0:
            pt_start_id = len(self.points)
            for k in range(self.N_select):
                self.points.append(Point(xyz=nan(3), id=pt_start_id + k))
            return False

        select_uv, select_depth = select_by_mask([self.uv, cur_depth], M_select, self.N_select)

        if self.use_uv_filter:
            filtered_idx = self.uv_filter(select_uv)
            select_uv = select_uv[filtered_idx, :]
            select_depth = select_depth[filtered_idx, :]

        P_c = unproj(select_uv, self.K, select_depth)  # (N_select, 3)
        P_w = transform(P_c, self.cameras[-1].T_wc)  # (N_select, 3)

        pt_start_id = len(self.points)
        for k in range(self.N_select):
            # maybe cannot select enough points
            if k >= P_w.shape[0]:
                self.points.append(Point(xyz=nan(3), id=pt_start_id + k))
                continue

            self.points.append(Point(xyz=P_w[k], id=pt_start_id + k))  # new 3D point
            cur_uv = select_uv[k]
            cur_point_id = k + cur_cam_id * self.N_select
            cur_point_window_id = cur_point_id - point_start_id

            self.obs.set_uv(cur_point_window_id, cur_cam_window_id, cur_uv)
            self.obs.set_sigma(cur_point_window_id, cur_cam_window_id, self.init_sigma)

            # (step 3) backward matching
            cur_uv_ = torch.from_numpy(cur_uv)[None, :, None, None].to(self.device)
            for j in range(-1, -self.window_size, -1):
                cam_window_id = cur_cam_window_id + j
                if cam_window_id < 0:
                    break

                backward_match = self.window_matches.backward_matches[j]
                can_match_backward = interpolate(cur_uv_, backward_match['mask'],
                                                 mode='nearest').squeeze().bool().item()
                if can_match_backward:

                    forward_match = self.window_matches.forward_matches[j]
                    flow_mask = flow_consistency_mask(cur_uv_, backward_match['flow'], forward_match['flow'],
                                                      th=self.th_flow)
                    if not interpolate(cur_uv_, flow_mask, mode='nearest').squeeze().bool().item():
                        break

                    backward_flow = interpolate(cur_uv_, backward_match['flow'])
                    last_uv = cur_uv_.squeeze().cpu().numpy()
                    cur_uv_ = cur_uv_ + backward_flow  # 2D position on j

                    if not interpolate(cur_uv_, forward_match['mask'], mode='nearest').squeeze().bool().item():
                        break

                    cur_uv = cur_uv_.squeeze().cpu().numpy()
                    if not is_in_image(cur_uv, self.height, self.width):
                        break

                    # epipolar check
                    T_cur_last = self.window_relative_poses.T_21[j]
                    s = epipolar_score(last_uv, cur_uv, T_cur_last, self.K, scale=1e3)
                    if s > self.th_score:
                        break

                    self.obs.set_uv(cur_point_window_id, cam_window_id, cur_uv)
                    self.obs.set_sigma(cur_point_window_id, cam_window_id, abs(j) * self.sigma2)

                else:
                    break

        return True

    def only_pose_bundle_adjustment(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(0)
        T_cw = np.linalg.inv(self.T_wc)
        T_cw = g2o.SE3Quat(T_cw[:3, :3], T_cw[:3, 3])
        v_se3.set_estimate(T_cw)
        v_se3.set_fixed(False)
        optimizer.add_vertex(v_se3)

        # only-pose-BA is for frame, because we have not added new keyframe yet
        # so the number of keyframes is len(self.cameras) - 1, plus current frame is len(self.cameras)
        cur_cam_id = len(self.cameras)
        start_point_id = (cur_cam_id - 1) * self.N_select
        end_point_id = cur_cam_id * self.N_select
        if start_point_id == 0:
            return

        for k in range(start_point_id, end_point_id):
            window_k = k - max((cur_cam_id - self.window_size) * self.N_select, 0)
            cam_start_id = max(cur_cam_id - self.window_size, 0)
            cur_cam_window_id = cur_cam_id - cam_start_id

            if not self.obs.is_valid(window_k, cur_cam_window_id - 1):
                continue

            edge = g2o.EdgeSE3ProjectXYZOnlyPose()
            edge.set_vertex(0, v_se3)

            # forward matching
            cur_uv = self.obs.matrix[window_k, cur_cam_window_id - 1]['uv']
            cur_uv_ = torch.from_numpy(cur_uv)[None, :, None, None].to(self.device)
            can_match_forward = interpolate(cur_uv_, self.M_total_forward, mode='nearest').squeeze().bool().item()
            if can_match_forward:
                forward_flow = interpolate(cur_uv_, self.forward_flow)
                cur_uv_ = cur_uv_ + forward_flow  # 2D position on cur

                can_match_backward = interpolate(cur_uv_, self.M_total_backward,
                                                 mode='nearest').squeeze().bool().item()
                if can_match_backward:
                    cur_uv = cur_uv_.squeeze().cpu().numpy()
                    if not is_in_image(cur_uv, self.height, self.width):
                        continue

                    sigma2 = self.obs.matrix[window_k, cur_cam_window_id - 1]['sigma2'] + self.sigma2
                    info = np.identity(2)
                    # info = info / sigma2
                    edge.set_measurement(cur_uv)
                    edge.set_information(info)

                    if self.robust_kernel:
                        rk = g2o.RobustKernelHuber()
                        rk.set_delta(math.sqrt(5.99))
                        edge.set_robust_kernel(rk)

                    edge.fx = self.K[0, 0]
                    edge.fy = self.K[1, 1]
                    edge.cx = self.K[0, 2]
                    edge.cy = self.K[1, 2]
                    edge.Xw = self.points[k].xyz

                    if self.points[k].is_bad():
                        edge.set_level(1)

                    edge.compute_error()
                    chi2 = edge.chi2()
                    is_depth_positive = edge.is_depth_positive()
                    if chi2 > 5.99 or not is_depth_positive:
                        edge.set_level(1)

                    optimizer.add_edge(edge)

        optimizer.initialize_optimization(0)
        optimizer.set_verbose(False)
        optimizer.optimize(10)

        # update states
        T_cw_ = optimizer.vertex(0).estimate().matrix().copy()
        self.T_wc = np.linalg.inv(T_cw_)
        self.T_12 = np.linalg.inv(self.cameras[-1].T_wc) @ self.T_wc

    def local_bundle_adjustment(self):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # EdgeSE3PointXYZDepth needs to input this, otherwise SIGSEGV
        cam = g2o.ParameterCamera()
        cam.set_id(0)
        optimizer.add_parameter(cam)

        cur_cam_id = len(self.cameras) - 1
        start_cam_id = max(cur_cam_id - self.window_size + 1, 0)
        cur_cam_window_id = cur_cam_id - start_cam_id
        for j in range(start_cam_id, cur_cam_id + 1):
            v_se3 = g2o.VertexSE3Expmap()
            cam_vertex_id = j - start_cam_id
            v_se3.set_id(cam_vertex_id)
            T_cw = self.cameras[j].T_cw
            T_cw = g2o.SE3Quat(T_cw[:3, :3], T_cw[:3, 3])
            v_se3.set_estimate(T_cw)

            # v1: self.num_overlap -> better
            # v2: self.window_size - self.num_overlap
            if cam_vertex_id < self.num_overlap:
                v_se3.set_fixed(True)
            optimizer.add_vertex(v_se3)

        # TODO: odom factor
        tmp1 = []
        for j in range(start_cam_id, cur_cam_id):
            cam1_vertex_id = j - start_cam_id
            cam2_vertex_id = j - start_cam_id + 1

            edge = g2o.EdgeSE3Expmap()
            edge.set_vertex(0, optimizer.vertex(cam1_vertex_id))
            edge.set_vertex(1, optimizer.vertex(cam2_vertex_id))
            T_21 = self.window_relative_poses.T_21[cam1_vertex_id]
            T_21 = g2o.SE3Quat(T_21[:3, :3], T_21[:3, 3])
            edge.set_measurement(T_21)

            T_21_gt = np.linalg.inv(self.traj_gt[self.keyframes[j + 1]]) @ self.traj_gt[self.keyframes[j]]
            delta_T = np.linalg.inv(T_21_gt) @ self.window_relative_poses.T_21[cam1_vertex_id]
            delta_pose = matrix_to_se3(delta_T)
            tmp1.append([np.abs(delta_pose[0]), np.abs(delta_pose[1]), np.abs(delta_pose[2]), np.abs(delta_pose[3]), np.abs(delta_pose[4]), np.abs(delta_pose[5])])

            info = np.identity(6)
            info[:3, :3] = info[:3, :3] * self.sigma2_odom_rot  # 1e6
            info[-3:, -3:] = info[-3:, -3:] * self.sigma2_odom_trans  # 10
            # edge.set_information(np.identity(6) * self.window_relative_poses.sigma2[cam1_vertex_id])
            # edge.set_information(info * self.sigma2_odom)
            edge.set_information(info)
            optimizer.add_edge(edge)

            # TODO: scale factor
            edge = g2o.EdgeTransNorm()
            edge.set_vertex(0, optimizer.vertex(cam1_vertex_id))
            edge.set_vertex(1, optimizer.vertex(cam2_vertex_id))
            T_12 = self.window_relative_poses.T_12[cam1_vertex_id]
            edge.delta_t = T_12[:3, 3]
            edge.set_information(np.eye(1) * self.sigma2_scale)
            optimizer.add_edge(edge)

        num1, num2 = 0, 0
        start_point_id = start_cam_id * self.N_select
        end_point_id = (cur_cam_id + 1) * self.N_select
        for k in range(start_point_id, end_point_id):
            vp = g2o.VertexPointXYZ()
            point_vertex_id = k - start_point_id + self.window_size
            vp.set_id(point_vertex_id)
            vp.set_marginalized(True)

            if self.points[k].is_bad():
                continue

            pt = self.points[k].xyz
            if np.isnan(pt).any():
                continue

            # do not add isolated points, which only has one observation in sliding window
            num = 0
            for j in range(0, cur_cam_window_id + 1):
                if self.obs.is_valid(k - start_point_id, j):
                    num = num + 1
            if num == 1:
                self.points[k].set_bad()
                continue

            vp.set_estimate(pt)
            optimizer.add_vertex(vp)

            for j in range(0, cur_cam_window_id + 1):
                cam_vertex_id = j
                if not self.obs.is_valid(k - start_point_id, j):
                    continue

                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, optimizer.vertex(cam_vertex_id))
                edge.set_measurement(self.obs.matrix[k - start_point_id, j]['uv'])
                sigma2 = self.obs.matrix[k - start_point_id, j]['sigma2']
                info = np.identity(2)
                # info = info / sigma2
                edge.set_information(info)
                if self.robust_kernel:
                    rk = g2o.RobustKernelHuber()
                    rk.set_delta(math.sqrt(5.99))
                    edge.set_robust_kernel(rk)

                edge.fx = self.K[0, 0]
                edge.fy = self.K[1, 1]
                edge.cx = self.K[0, 2]
                edge.cy = self.K[1, 2]
                num1 = num1 + 1

                edge.compute_error()
                chi2 = edge.chi2()
                is_depth_positive = edge.is_depth_positive()
                if chi2 > 5.99 or not is_depth_positive:
                    num2 = num2 + 1
                    edge.set_level(1)
                    point_id = start_point_id + edge.vertex(0).id() - self.window_size
                    self.points[point_id].set_bad()
                    optimizer.remove_vertex(edge.vertex(0))

                optimizer.add_edge(edge)

        optimizer.initialize_optimization(0)
        optimizer.set_verbose(False)
        optimizer.optimize(32)
        # print('discard ratio in BA: {} / {} = {:.2f} %'.format(num2, num1, num2 / num1 * 100))

        # update states
        for j in range(start_cam_id, cur_cam_id + 1):
            cam_vertex_id = j - start_cam_id
            if cam_vertex_id == 0:
                continue
            T_cw_ = optimizer.vertex(cam_vertex_id).estimate().matrix().copy()
            self.cameras[j].set_from_Tcw(T_cw_)

        for k in range(start_point_id, end_point_id):
            point_vertex_id = k - start_point_id + self.window_size
            if optimizer.vertex(point_vertex_id) is None:
                continue
            xyz_ = optimizer.vertex(point_vertex_id).estimate().copy()
            self.points[k].set_xyz(xyz_)

    def do_local_ba(self):
        condition_1 = (len(self.keyframes) >= self.window_size)

        condition_2 = False
        is_first_window = (len(self.keyframes) == self.window_size)
        if is_first_window:
            condition_2 = True
        elif self.cnt_to_last_ba == (self.window_size - self.num_overlap):
            condition_2 = True
            self.cnt_to_last_ba = 0

        return condition_1 and condition_2

    @torch.no_grad()
    def run(self):
        with Timer('MaskFlow-SVO', enabled=True, N=len(self.dataset)):
            for i, data in tqdm(enumerate(self.dataset), total=len(self.dataset) - 1, desc='Seq. {}: '.format(self.seq)):
                # ---------------------------- Prepare Data ----------------------------
                cur_frame_id = i + 1  # i start from zero (i.e. first pair), note to plus the first frame

                left_img1 = data.frame1.left_img.unsqueeze(0).to(self.device)  # (B, 3, H, W)
                right_img1 = data.frame1.right_img.unsqueeze(0).to(self.device)  # (B, 3, H, W)
                left_img2 = data.frame2.left_img.unsqueeze(0).to(self.device)  # (B, 3, H, W)
                right_img2 = data.frame2.right_img.unsqueeze(0).to(self.device)  # (B, 3, H, W)
                left_K = data.frame1.left_K.unsqueeze(0).to(self.device)  # (B, 3, 3)
                right_K = data.frame1.right_K.unsqueeze(0).to(self.device)  # (B, 3, 3)
                K = 0.5 * (left_K + right_K)
                intrinsic_layer = data.intrinsic_layer.unsqueeze(0).to(self.device)  # (B, 2, H, W)

                flow_l1l2 = data.flow_l1l2.unsqueeze(0).to(self.device)
                flow_l2l1 = data.flow_l2l1.unsqueeze(0).to(self.device)
                flows = torch.cat((flow_l1l2, flow_l2l1), dim=1)  # (B, 4, H, W)

                disp1 = data.disp1.unsqueeze(0).to(self.device)
                disp2 = data.disp2.unsqueeze(0).to(self.device)
                disps = torch.cat((disp1, disp2), dim=1)  # (B, 2, H, W)

                prior_mask1 = data.mask1.unsqueeze(0).to(self.device)
                prior_mask2 = data.mask2.unsqueeze(0).to(self.device)
                prior_masks = torch.cat((prior_mask1, prior_mask2), dim=1)  # (B, 20*2, H, W)

                # ---------------------------- Pose Network ----------------------------
                rot, trans, confidence, \
                    flow_l1l2, M_neural_forward, \
                    flow_l2l1, M_neural_backward = \
                    self.network(flow=flows, disps=disps, prior_masks=prior_masks,
                                 K=K, intrinsic_layer=intrinsic_layer, backward=True)

                pred_rot = rot  # (B, 3)
                baseline = data.frame1.baseline[None, None].to(self.device)  # (B, 1)
                pred_trans = trans * baseline  # (B, 3)

                pred_pose = torch.cat((pred_rot, pred_trans), dim=-1)  # (B, 6)
                pred_pose_np = pred_pose.squeeze(0).cpu().numpy()  # (6)
                T_12 = se3_to_matrix(pred_pose_np)  # (4, 4)
                T_cur_frontend = self.traj_frontend[-1] @ T_12  # T_wc

                transform_12 = data.transform_12
                pose_gt_np = torch.cat((transform_12.so3, transform_12.t)).numpy()
                T_12_gt = se3_to_matrix(pose_gt_np)  # (4, 4)
                T_cur_gt = self.traj_gt[-1] @ T_12_gt

                self.timestamps.append(data.frame2.timestamp)
                self.traj_frontend.append(T_cur_frontend)
                self.traj_gt.append(T_cur_gt)

                self.T_12 = self.T_12 @ T_12
                self.T_12_net = self.T_12_net @ T_12
                self.T_wc = self.cameras[-1].T_wc @ self.T_12

                if not self.args.ba:
                    continue

                # ---------------------------- Slinding Window Optimization ----------------------------
                self.history_forward_masks.append(M_neural_forward)

                # forward correspondences
                depth1 = disp_to_depth_v2(disp1, K[:, 0, 0], baseline)  # (B, 1, H, W)
                self.compute_forward_corr(flow_l1l2, flow_l2l1, depth1, M_neural_forward)

                # backward correspondences
                depth2 = disp_to_depth_v2(disp2, K[:, 0, 0], baseline)  # (B, 1, H, W)
                self.compute_backward_corr(flow_l2l1, flow_l1l2, depth2, M_neural_backward)

                # TODO: visualization
                # self.vis_match(cur_img=left_img2)

                # save useful variables
                self.K = left_K.squeeze(0).cpu().numpy()  # (3, 3)
                self.bf = self.K[0, 0] * baseline.item()
                self.cur_disp = torch.abs(disp2).clamp(min=1e-6)  # (B, 1, H, W)
                self.cache_M_neural_forward.append(M_neural_forward)
                self.cache_M_neural_backward.append(M_neural_backward)
                if cur_frame_id == 1:
                    first_img = left_img1.squeeze(0).cpu().numpy()  # (3, H, W)
                    first_img = np.array(first_img.transpose(1, 2, 0) * 255, dtype=np.uint8)  # (H, W, 3)
                    self.window_keyframes.save_images(first_img)
                    disp1_np = torch.abs(disp1).clamp(min=1e-6).squeeze(0).cpu().numpy()  # (1, H, W)
                    self.window_keyframes.save_disps(disp1_np)
                    self.window_keyframes.save_depths(depth1)

                with Timer("Only Pose Bundle Adjustment", enabled=self.enable_timing):
                    self.only_pose_bundle_adjustment()

                # backend
                # TODO: not yet parallelized in another thread
                if self.is_keyframe(cur_frame_id):
                    cur_img = left_img2.squeeze(0).cpu().numpy()  # (3, H, W)
                    cur_img = np.array(cur_img.transpose(1, 2, 0) * 255, dtype=np.uint8)  # (H, W, 3)
                    self.window_keyframes.save_images(cur_img)
                    disp2_np = self.cur_disp.squeeze(0).cpu().numpy()  # (1, H, W)
                    self.window_keyframes.save_disps(disp2_np)
                    self.window_keyframes.save_depths(depth2)

                    with Timer("Expand Graph", enabled=self.enable_timing):
                        if not self.expand_graph(self.T_wc, depth2):
                            continue
                        else:
                            pass

                    # bundle adjustment in sliding window
                    if self.do_local_ba():
                        with Timer("Local Bundle Adjustment", enabled=self.enable_timing):
                            self.local_bundle_adjustment()

                        # TODO: visualization
                        # self.vis_matches()

        # ---------------------------- Evaluation ----------------------------
        self.save_trajectory(interpolate_kf=True)

    def vis_match(self, cur_img):
        keyframe1_id = self.keyframes[-1]
        if keyframe1_id == 0:
            keyframe1 = self.dataset[keyframe1_id].frame1.left_img.unsqueeze(0).to(self.device)
        else:
            keyframe1 = self.dataset[keyframe1_id].frame2.left_img.unsqueeze(0).to(self.device)
        keyframe1_np = np.array(keyframe1[0].permute(1, 2, 0).cpu().detach() * 255, dtype=np.uint8)
        M_flow_total_forward_np = np.array(self.M_flow_total_forward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                           dtype=np.uint8) * 255
        M_depth_total_forward_np = np.array(self.M_depth_total_forward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                            dtype=np.uint8) * 255
        M_epi_total_forward_np = np.array(self.M_epi_total_forward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                          dtype=np.uint8) * 255
        M_neural_total_forward_np = np.array(self.M_neural_total_forward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                             dtype=np.uint8) * 255
        M_total_forward_np = np.array(self.M_total_forward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                      dtype=np.uint8) * 255
        img_mask_forward = cv2.vconcat(
            [M_flow_total_forward_np, M_epi_total_forward_np, M_neural_total_forward_np, M_total_forward_np])

        cur_img1_np = np.array(cur_img[0].permute(1, 2, 0).cpu().detach() * 255, dtype=np.uint8)
        M_flow_total_backward_np = np.array(self.M_flow_total_backward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                            dtype=np.uint8) * 255
        M_depth_total_backward_np = np.array(self.M_depth_total_backward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                             dtype=np.uint8) * 255
        M_epi_total_backward_np = np.array(self.M_epi_total_backward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                           dtype=np.uint8) * 255
        M_neural_total_backward_np = np.array(self.M_neural_total_backward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                              dtype=np.uint8) * 255
        M_total_backward_np = np.array(self.M_total_backward[0].permute(1, 2, 0).repeat(1, 1, 3).cpu(),
                                       dtype=np.uint8) * 255
        img_mask_backward = cv2.vconcat(
            [M_flow_total_backward_np, M_epi_total_backward_np, M_neural_total_backward_np, M_total_backward_np])

        u_set = torch.masked_select(self.uv[0, 0], self.M_total_forward[0, 0].bool())  # (N)
        v_set = torch.masked_select(self.uv[0, 1], self.M_total_forward[0, 0].bool())  # (N)
        assert u_set.shape[0] == v_set.shape[0]
        set_id = torch.randperm(u_set.shape[0])[:self.N_select]
        u1_set = u_set[set_id]
        v1_set = v_set[set_id]
        uv1_set = torch.stack((u1_set, v1_set), dim=-1)  # (N_select, 2)
        uv1_set_np = uv1_set.cpu().numpy()

        forward_flowu_set = torch.masked_select(self.forward_flow[0, 0], self.M_total_forward[0, 0].bool())  # (N)
        forward_flowv_set = torch.masked_select(self.forward_flow[0, 1], self.M_total_forward[0, 0].bool())  # (N)
        forward_flowu_set = forward_flowu_set[set_id]
        forward_flowv_set = forward_flowv_set[set_id]
        forward_flow_set = torch.stack((forward_flowu_set, forward_flowv_set), dim=-1)  # (N_select, 2)
        uv2_set = uv1_set + forward_flow_set  # (N_select, 2)
        uv2_set_np = uv2_set.cpu().numpy()

        kp1, kp2, matches = [], [], []
        for k in range(self.N_select):
            kp1.append(cv2.KeyPoint(x=uv1_set_np[k, 0], y=uv1_set_np[k, 1], size=1))
            kp2.append(cv2.KeyPoint(x=uv2_set_np[k, 0], y=uv2_set_np[k, 1], size=1))
            matches.append(cv2.DMatch(_queryIdx=k, _trainIdx=k, _distance=0))

        im_matches = cv2.drawMatches(keyframe1_np, kp1, cur_img1_np, kp2, matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                     matchColor=[0, 255, 0])

        img_mask = cv2.hconcat([img_mask_forward, img_mask_backward])
        img_mask = cv2.vconcat([im_matches, img_mask])
        cv2.imshow('mask', img_mask)
        cv2.setWindowTitle('mask', '|avg_flow|: {:.2f}'.format(self.avg_flow_norm))
        cv2.waitKey(0)

    def vis_matches(self):
        cur_cam_id = len(self.cameras) - 1
        start_cam_id = max(cur_cam_id - self.window_size + 1, 0)
        kf_img1 = self.window_keyframes.images[start_cam_id - cur_cam_id - 1]
        kf_img1 = cv2.rotate(kf_img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # print('\n', self.obs.matrix)

        for j in range(start_cam_id, cur_cam_id + 1):
            cam1_id = j
            cam2_id = j + 1
            cam1_window_id = j - start_cam_id
            cam2_window_id = j + 1 - start_cam_id
            w0 = (j - start_cam_id) * self.height
            kf_img2 = self.window_keyframes.images[cam2_id - cur_cam_id - 1]
            kf_img2 = cv2.rotate(kf_img2, cv2.ROTATE_90_COUNTERCLOCKWISE)

            num_matches = 0
            kp1, kp2, matches = [], [], []
            for point_id in range(start_cam_id * self.N_select, (cur_cam_id + 1) * self.N_select):
                pt = self.points[point_id]
                if pt.is_bad():
                    continue

                point_window_id = point_id - start_cam_id * self.N_select

                # draw re-projection error
                if self.obs.is_valid(point_window_id, cam1_window_id):
                    uv1 = self.obs.matrix[point_window_id, cam1_window_id]['uv']
                    pt_in_cam = transform(pt.xyz[None, :], self.cameras[cam1_id].T_cw)
                    uv1_proj = proj(pt_in_cam, self.K).squeeze()

                    cv2.arrowedLine(img=kf_img1,
                                    pt1=(round(uv1[1] + w0), round(self.width - uv1[0] - 1)),
                                    pt2=(round(uv1_proj[1] + w0), round(self.width - uv1_proj[0] - 1)),
                                    color=(0, 0, 255), thickness=2)

                if cam2_window_id == self.window_size:
                    continue

                # draw match
                if self.obs.is_valid(point_window_id, cam1_window_id) and self.obs.is_valid(point_window_id, cam2_window_id):
                    uv1 = self.obs.matrix[point_window_id, cam1_window_id]['uv']
                    uv2 = self.obs.matrix[point_window_id, cam2_window_id]['uv']
                    kp1.append(cv2.KeyPoint(x=uv1[1] + w0, y=self.width - uv1[0] - 1, size=1))
                    kp2.append(cv2.KeyPoint(x=uv2[1], y=self.width - uv2[0] - 1, size=1))
                    matches.append(cv2.DMatch(_queryIdx=num_matches, _trainIdx=num_matches, _distance=0))
                    num_matches = num_matches + 1

            if cam2_window_id < self.window_size:
                kf_img1 = cv2.drawMatches(kf_img1, kp1, kf_img2, kp2, matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                                          matchColor=[0, 255, 0])

        kf_img1 = cv2.rotate(kf_img1, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('image matches', kf_img1)
        cv2.setWindowTitle('image matches', '{}'.format(len(self.cameras) - self.window_size))
        cv2.waitKey(0)

    def interpolate_kf_traj(self):
        assert len(self.keyframes) > 1

        timestamps_backend, traj_backend = [], []
        kf_1_id, i0 = 0, 0
        for i, timestamp in tqdm(enumerate(self.timestamps), total=len(self.timestamps) - 1,
                                 desc='Interpolating keyframe poses: '):
            kf_1 = self.timestamps[self.keyframes[kf_1_id]]
            kf_2 = self.timestamps[self.keyframes[kf_1_id + 1]]

            if timestamp == kf_1:
                timestamps_backend.append(timestamp)
                traj_backend.append(self.cameras[kf_1_id].T_wc)

            elif kf_1 < timestamp < kf_2:
                pose_kf_1 = self.cameras[kf_1_id].T_wc
                pose_kf_2 = self.cameras[kf_1_id + 1].T_wc

                key_times = [kf_1, kf_2]
                key_rots = R.from_matrix([pose_kf_1[:3, :3], pose_kf_2[:3, :3]])
                slerp = Slerp(key_times, key_rots)
                times = [timestamp]
                interp_rots = slerp(times)
                interp_rot = interp_rots.as_matrix()[0]

                left_linear_weight = (kf_2 - timestamp) / (kf_2 - kf_1)
                right_linear_weight = 1 - left_linear_weight
                interp_trans = left_linear_weight * pose_kf_1[:3, 3] + right_linear_weight * pose_kf_2[:3, 3]

                T_wb = np.eye(4)
                T_wb[:3, :3] = interp_rot
                T_wb[:3, 3] = interp_trans

                timestamps_backend.append(timestamp)
                traj_backend.append(T_wb)

            elif timestamp == kf_2:
                timestamps_backend.append(timestamp)
                traj_backend.append(self.cameras[kf_1_id + 1].T_wc)
                i0 = i

                if kf_2 < self.timestamps[self.keyframes[-1]]:
                    kf_1_id = kf_1_id + 1

            if timestamp > self.timestamps[self.keyframes[-1]]:
                assert kf_2 == self.timestamps[self.keyframes[-1]]
                T_12 = np.linalg.inv(self.traj_frontend[i0]) @ self.traj_frontend[i]
                T_cur = self.cameras[kf_1_id + 1].T_wc @ T_12
                timestamps_backend.append(timestamp)
                traj_backend.append(T_cur)

        return timestamps_backend, traj_backend

    def save_trajectory(self, use_raw_gt=False, interpolate_kf=False):
        save_dir = os.path.join('./results', self.dataset_name, self.seq)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamps_gt = self.raw_timestamps if use_raw_gt else self.timestamps
        traj_gt = self.traj_raw_gt if use_raw_gt else self.traj_gt

        if len(self.keyframes) == 1:
            print('Changing interpolate_kf to False! (len_kf: {})'.format(len(self.keyframes)))
            interpolate_kf = False

        if interpolate_kf:
            # print('before: {}'.format(len(self.cameras)))
            timestamps_backend, traj_backend = self.interpolate_kf_traj()
            # print('after: {}'.format(len(traj_backend)))
            # print('groundtruth: {}'.format(len(self.traj_gt)))

        ###################
        # save TUM format #
        ###################
        est_path_tum = os.path.join(save_dir, 'est_tum.txt')
        est_kf_path_tum = os.path.join(save_dir, 'est_kf_tum.txt')
        gt_path_tum = os.path.join(save_dir, 'gt_tum.txt')

        # save file in TUM format (timestamp tx ty tz qx qy qz qw)
        # which can be evaluated by evo tools
        pose_to_save = []
        for timestamp, T in zip(self.timestamps, self.traj_frontend):
            r = R.from_matrix(T[0:3, 0:3])
            q_12 = r.as_quat().astype(np.float64)  # (x, y, z, w)
            t_12 = np.array(T[0:3, 3]).astype(np.float64)
            pose_to_save.append(np.concatenate((timestamp, t_12, q_12), axis=None))
        np.savetxt(est_path_tum, np.array(pose_to_save))
        print(f'\033[1;32m [success] \033[0m', '(TUM) Save estimated trajectory to {}'.format(est_path_tum))

        pose_gt_to_save = []
        for timestamp, T_gt in zip(timestamps_gt, traj_gt):
            r_gt = R.from_matrix(T_gt[0:3, 0:3])
            q_12_gt = r_gt.as_quat().astype(np.float64)
            t_12_gt = np.array(T_gt[0:3, 3]).astype(np.float64)
            pose_gt_to_save.append(np.concatenate((timestamp, t_12_gt, q_12_gt), axis=None))
        np.savetxt(gt_path_tum, np.array(pose_gt_to_save))
        print(f'\033[1;32m [success] \033[0m', '(TUM) Save groundtruth trajectory to {}'.format(gt_path_tum))

        kf_pose_to_save = []
        if interpolate_kf:
            for timestamp, T in zip(timestamps_backend, traj_backend):
                r = R.from_matrix(T[0:3, 0:3])
                q_12 = r.as_quat().astype(np.float64)  # (x, y, z, w)
                t_12 = np.array(T[0:3, 3]).astype(np.float64)
                kf_pose_to_save.append(np.concatenate((timestamp, t_12, q_12), axis=None))
        else:
            for j, camera in enumerate(self.cameras):
                timestamp = self.timestamps[self.keyframes[j]]
                Twc = camera.T_wc
                r = R.from_matrix(Twc[0:3, 0:3])
                q_wc = r.as_quat().astype(np.float64)  # (x, y, z, w)
                t_wc = np.array(Twc[0:3, 3]).astype(np.float64)
                kf_pose_to_save.append(np.concatenate((timestamp, t_wc, q_wc), axis=None))
        np.savetxt(est_kf_path_tum, np.array(kf_pose_to_save))
        print(f'\033[1;32m [success] \033[0m', '(TUM) Save estimated keyframe trajectory to {}'.format(est_kf_path_tum))

        print(f'\033[1;33m [Traj] \033[0m',
              'run:  evo_traj tum --ref {} {} {} --plot_mode xz -vp'.format(gt_path_tum,
                                                                            est_path_tum,
                                                                            est_kf_path_tum))

        #####################
        # save KITTI format #
        #####################
        est_path_kitti = os.path.join(save_dir, 'est_kitti.txt')
        est_kf_path_kitti = os.path.join(save_dir, 'est_kf_kitti.txt')
        gt_path_kitti = os.path.join(save_dir, 'gt_kitti.txt')

        pose_to_save = []
        for timestamp, T_wc in zip(self.timestamps, self.traj_frontend):
            row = T_wc[:3].flatten().astype(np.float64)  # 1 x 12
            pose_to_save.append(np.concatenate((timestamp, row), axis=None))
        np.savetxt(est_path_kitti, np.array(pose_to_save))
        print(f'\033[1;32m [success] \033[0m', '(KITTI) Save estimated trajectory to {}'.format(est_path_kitti))

        pose_gt_to_save = []
        for timestamp, T_wc_gt in zip(timestamps_gt, traj_gt):
            row_gt = T_wc_gt[:3].flatten().astype(np.float64)  # 1 x 12
            pose_gt_to_save.append(np.concatenate((timestamp, row_gt), axis=None))
        np.savetxt(gt_path_kitti, np.array(pose_gt_to_save))
        print(f'\033[1;32m [success] \033[0m', '(KITTI) Save groundtruth trajectory to {}'.format(gt_path_kitti))

        kf_pose_to_save = []
        if interpolate_kf:
            for timestamp, T_wc in zip(timestamps_backend, traj_backend):
                row = T_wc[:3].flatten().astype(np.float64)  # 1 x 12
                kf_pose_to_save.append(np.concatenate((timestamp, row), axis=None))
        else:
            for j, camera in enumerate(self.cameras):
                timestamp = self.timestamps[self.keyframes[j]]
                T_wc = camera.T_wc
                row_kf = T_wc[:3].flatten().astype(np.float64)  # 1 x 12
                kf_pose_to_save.append(np.concatenate((timestamp, row_kf), axis=None))
        np.savetxt(est_kf_path_kitti, np.array(kf_pose_to_save))
        print(f'\033[1;32m [success] \033[0m',
              '(KITTI) Save estimated keyframe trajectory to {}'.format(est_kf_path_kitti))

        t_rel, r_rel, ate, rpe_trans, rpe_rot = self.eval_tool.eval(gt_path_kitti, est_path_kitti, quiet=True)
        t_rel_kf, r_rel_kf, ate_kf, rpe_trans_kf, rpe_rot_kf = self.eval_tool.eval(gt_path_kitti, est_kf_path_kitti, quiet=True)

        output_path = './results/output.txt'
        with open(output_path, 'a') as f:
            f.write('{} {} {:.2f} {:.2f} {:.2f}  ||  '.format(self.dataset_name, self.seq, t_rel, r_rel, ate))
            f.write('{:.2f} {:.2f} {:.2f}\n'.format(t_rel_kf, r_rel_kf, ate_kf))


def main(args):
    slam = SLAM(args)
    slam.run()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='kitti_tracking', choices=['kitti', 'kitti_tracking'])
    parser.add_argument('--model', type=str, default='./configs/model_predict.yaml')
    parser.add_argument('--seq', type=str, default=None)
    parser.add_argument('--ba', type=bool, default=True)
    parser.add_argument('--kitti_submit', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    args.dataset = './configs/{}.yaml'.format(args.dataset_name)

    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    np.set_printoptions(linewidth=200)

    main(args)

    print('Done.')
