import argparse

import cv2
import torch
from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation

import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.models as models
from pytorch3d.transforms.so3 import so3_exp_map
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = "tight"

# optical flow
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from utils import *

sys.path.append('/home/xiechen/科研/E2E_stereo_VO/v3/models')
from models.swin_unet import SwinPoseNet


class MaskedPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.normalize = cfg['model'].network.normalize
        self.intrinsic_layer = cfg['model'].network.intrinsic_layer
        self.beta = cfg['model'].network.beta
        self.height = cfg['model'].network.height
        self.width = cfg['model'].network.width
        self.th_conf = 0.5

        # #############################################
        # RAFT
        self.weights = Raft_Large_Weights.DEFAULT
        self.raft = raft_large(weights=self.weights, progress=False)
        # fix the weight
        for param in self.raft.parameters():
            param.requires_grad = False

        #############################################
        # PoseNet: to be trained
        self.pose_net = SwinPoseNet(
            channels=5,  # <- TODO
            input_size=(self.height, self.width),
            hidden_dim=32,  # <- TODO
            layers=(2, 2, 2),
            heads=(3, 6, 9),
            head_dim=32,
            window_size=(8, 8, 4),  # window_size 要整除下采样后的尺寸
            downscaling_factors=(4, 2, 2),  # (192, 640) | /4-> (48, 160) /2-> (24, 80) /2-> (12, 40)
            relative_pos_embedding=True,
            multi_scale=True,
            th_conf=self.th_conf,
            ln_layer=False  # <- TODO
        )

    def forward(self, flow,
                disps, prior_masks,
                K, intrinsic_layer, backward=False):
        """
        flow: (B, 2/4, H, W), depend on backward
        disps: (B, 1/2, H, W), depend on backward
        prior_masks: (B, N, H, W) or None
        """
        device = flow.device
        with torch.no_grad():
            # record original flow, used later to estimate mask
            raw_flow = flow.clone()
            raw_flow_l1l2 = raw_flow[:, 0:2, ...]
            if backward:
                raw_flow_l2l1 = raw_flow[:, 2:4, ...]

            # normalize: (36.000us 10.375MB)
            b, _, h, w = flow.shape
            if self.normalize:
                flow = flow.reshape(b, -1, 2, h, w)  # (B, n, 2, H, W), verified
                flow_u = flow[:, :, 0, ...]  # (B, 1/2, H, W)
                flow_v = flow[:, :, 1, ...]  # (B, 1/2, H, W)
                flow_u = torch.div(flow_u, K[:, 0, 0][:, None, None, None]).unsqueeze(2) * self.beta
                flow_v = torch.div(flow_v, K[:, 1, 1][:, None, None, None]).unsqueeze(2) * self.beta
                flow = torch.cat((flow_u, flow_v), dim=2)  # (B, n, 2, H, W)
                flow = flow.reshape(b, -1, h, w)  # (B, 2/4, H, W)

                disps = torch.div(disps, K[:, 0, 0][:, None, None, None]) * self.beta  # (B, 1/2, H, W)

            flow_l1l2 = flow[:, 0:2, ...]  # (B, 2, H, W)
            disp1 = torch.abs(disps[:, 0, ...].unsqueeze(1)).clamp(min=1e-6)  # (B, 1, H, W)

            if backward:
                flow_l2l1 = flow[:, 2:4, ...]  # (B, 2, H, W)
                disp2 = torch.abs(disps[:, 1, ...].unsqueeze(1)).clamp(min=1e-6)  # (B, 1, H, W)

            flow = torch.cat((flow_l1l2, disp1), dim=1)  # (B, 3, H, W)
            if self.intrinsic_layer:
                flow = torch.cat((flow, intrinsic_layer), dim=1)  # (B, 5, H, W)

            if backward:
                back_flow = torch.cat((flow_l2l1, disp2), dim=1)  # (B, 3, H, W)
                if self.intrinsic_layer:
                    back_flow = torch.cat((back_flow, intrinsic_layer), dim=1)  # (B, 5, H, W)

        # predicted pose and mask
        if self.training:
            pose, confidence, final_mask = self.pose_net(flow)
        else:
            pose, confidence, final_mask = self.pose_net(flow, prior_masks=prior_masks[:, 0:20, ...])

        rotation = pose[:, :3]
        translation = pose[:, -3:]

        if backward:
            if self.training:
                _, back_confidence, final_mask_back = self.pose_net(back_flow)
            else:
                _, back_confidence, final_mask_back = self.pose_net(back_flow, prior_masks=prior_masks[:, 20:40, ...])

            return rotation, translation, confidence, raw_flow_l1l2, final_mask, raw_flow_l2l1, final_mask_back

        return rotation, translation, confidence, raw_flow_l1l2, final_mask


if __name__ == '__main__':
    print('Done.')
