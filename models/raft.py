import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as F2
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = "tight"

# optical flow
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image

from utils import *


class RAFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = Raft_Large_Weights.DEFAULT
        self.raft = raft_large(weights=self.weights, progress=False)

    def forward(self, x, backward=False):
        # image: I_{t-1} and I_{t}
        img = {
            'img1': x[:, :3],
            'img2': x[:, 3:6]
        }

        # BGR to RGB
        img['img1'] = img['img1'][:, [2, 1, 0], ...]
        img['img2'] = img['img2'][:, [2, 1, 0], ...]

        b, c, h, w = img['img1'].shape

        # flow
        img1_ = F2.normalize(img['img1'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous()
        img2_ = F2.normalize(img['img2'], mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).contiguous()

        if backward:
            img1 = torch.cat((img1_.unsqueeze(1), img2_.unsqueeze(1)), dim=1).reshape(-1, c, h, w).contiguous()
            img2 = torch.cat((img2_.unsqueeze(1), img1_.unsqueeze(1)), dim=1).reshape(-1, c, h, w).contiguous()
            flow = self.raft(img1, img2)[-1]  # (B, 2, H, W)
            flow = flow.reshape(b, -1, 2, h, w)  # (B, N, 2, H, W)

            forward_flow = flow[:, 0]  # (B, 2, H, W)
            backward_flow = flow[:, 1]  # (B, 2, H, W)
            return forward_flow, backward_flow

        else:
            img1 = img1_
            img2 = img2_
            flow = self.raft(img1, img2)[-1]  # (B, 2, H, W)
            return flow
