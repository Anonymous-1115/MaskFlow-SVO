import os
import re
import sys
import uuid
import math
import copy
import time
import yaml
import random
import numpy as np
from scipy import misc
import torch
from torch import nn
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
from PIL import Image
import PIL.Image as pil
from PIL import Image, ImageDraw, ImageFont
import matplotlib as mpl
import matplotlib.cm as cm

from scipy.spatial.transform import Rotation as R2
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp

import torch.nn.functional as F
import torchvision.transforms.functional as F2
from pytorch3d.transforms.so3 import so3_exp_map
from contextlib import ContextDecorator
from scipy import interpolate as inter
from matplotlib.ticker import MaxNLocator
import kornia
from einops import rearrange


def set_seed(seed=42):
    """
    一键设置所有随机种子，以确保结果可复现。
    参数:
        seed (int): 随机种子，默认为42。
    """
    print('Set seed to {}'.format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 针对多GPU情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cv2_show_image(image: torch.Tensor, shape='hwc'):
    """
    注：如果黑屏了就检查一下 shape 参数有没有写
    """
    if shape == 'bchw':
        image = image[0].permute(1, 2, 0)
        shape = 'hwc'
    elif shape == 'chw':
        image = image.permute(1, 2, 0)
        shape = 'hwc'
    assert shape == 'hwc'

    image = image[:, :, [2, 1, 0]].cpu()

    if image.max() == 1:
        cv2.imshow('img', np.array(image * 255, dtype=np.uint8))
        cv2.waitKey(0)
    else:
        cv2.imshow('img', np.array(image, dtype=np.uint8))
        cv2.waitKey(0)


def cv2_show_depth(depth: torch.Tensor, min_depth=None, max_depth=None, shape='hwc'):
    """
    注：如果黑屏了就检查一下 shape 参数有没有写
    """
    depth = torch.abs(depth.cpu())

    if shape == 'bchw':
        depth = depth[0].permute(1, 2, 0)
        shape = 'hwc'
    elif shape == 'chw':
        depth = depth.permute(1, 2, 0)
        shape = 'hwc'
    assert shape == 'hwc'

    if max_depth is None:
        max_depth = depth.max()

    if min_depth is None:
        min_depth = depth.min()

    depth = (depth - min_depth) / max(max_depth - min_depth, 1e-4) * 255
    depth = torch.clamp(depth, min=0)  # 有的 mask 是 0，此时是负数，需要显示黑色

    cv2.imshow('depth', np.array(depth, dtype=np.uint8))
    cv2.waitKey(0)


def cv2_show_image_and_depth(image: torch.Tensor, depth: torch.Tensor, min_depth=None, max_depth=None,
                             shape='hwc', text='img_and_depth', bgr=True):
    image = image.cpu()
    depth = torch.abs(depth).cpu()

    if shape == 'bchw':
        image = image[0].permute(1, 2, 0)
        depth = depth[0].permute(1, 2, 0)
        shape = 'hwc'
    elif shape == 'chw':
        image = image.permute(1, 2, 0)
        depth = depth.permute(1, 2, 0)
        shape = 'hwc'
    assert shape == 'hwc'

    if bgr:
        image = image[:, :, [2, 1, 0]]

    if max_depth is None:
        max_depth = depth.max()

    if min_depth is None:
        min_depth = depth.min()

    if min_depth == max_depth and min_depth == 1:
        depth = torch.ones_like(depth) * 255
    elif min_depth == max_depth and min_depth == 0:
        depth = torch.zeros_like(depth) * 255
    else:
        depth = (depth - min_depth) / (max_depth - min_depth) * 255
        depth = torch.clamp(depth, min=0)  # 有的 mask 是 0，此时是负数，需要显示黑色

    depth_output = np.array(depth, dtype=np.uint8)

    if math.isclose(image.max().item(), 1, rel_tol=1e-4):
        image_output = np.array(image * 255, dtype=np.uint8)
    else:
        image_output = np.array(image, dtype=np.uint8)

    output = np.vstack((image_output, depth_output.repeat(3, -1)))
    cv2.imshow('img_and_depth', output)
    cv2.setWindowTitle('img_and_depth', text)
    cv2.waitKey(0)


def cv2_save_img(image: torch.Tensor, shape='hwc', path='./img.jpg', quiet=False):
    if shape == 'bchw':
        image = image[0].permute(1, 2, 0)
        shape = 'hwc'
    elif shape == 'chw':
        image = image.permute(1, 2, 0)
        shape = 'hwc'
    assert shape == 'hwc'

    cv2.imwrite(path, np.array(image, dtype=np.uint8))
    if not quiet:
        print(f'\033[1;32m [success] \033[0m', 'save image to {}'.format(path))


def cv2_save_depth(depth: torch.Tensor, shape='hwc', path='./depth.jpg', min_depth=None, max_depth=None,
                   text=None, text_size=30, quiet=False):
    if shape == 'bchw':
        depth = depth[0].permute(1, 2, 0)
        shape = 'hwc'
    elif shape == 'chw':
        depth = depth.permute(1, 2, 0)
        shape = 'hwc'
    assert shape == 'hwc'

    if max_depth is None:
        max_depth = depth.max()

    if min_depth is None:
        min_depth = depth.min()

    if min_depth == max_depth and min_depth == 0:
        depth = torch.ones_like(depth) * 0
    elif min_depth == max_depth and min_depth != 0:
        depth = torch.ones_like(depth) * 255
    else:
        depth = (depth - min_depth) / (max_depth - min_depth) * 255

    # TODO
    im = pil.fromarray(np.uint8(depth.repeat(1, 1, 3).numpy()))
    if text is not None:
        im = pil_image_add_text(im, text, left=0, top=0, text_color=(0, 255, 0), text_size=text_size)
    im.save(path)
    # cv2.imwrite(path, np.array(depth, dtype=np.uint8))

    if not quiet:
        print(f'\033[1;32m [success] \033[0m', 'save depth to {}'.format(path))


def angle_difference(delta_R, delta_R_gt, degree=True):
    res = np.arccos((np.trace(delta_R @ delta_R_gt.transpose()) - 1) / 2)
    if degree is True:
        res = res * 180 / np.pi
    return res


def angle_difference_v2(rot, rot_gt, degree=True):
    '''

    angle difference = acos((tr(R @ R_gt.t) - 1) / 2)

    Args:
        rot: (B, 3)
        rot_gt: (B, 3)
        degree: bool

    Returns: (B)

    '''
    R = so3_exp_map(rot)  # (B, 3, 3)
    R_gt = so3_exp_map(rot_gt)  # (B, 3, 3)

    tmp = torch.matmul(R, torch.transpose(R_gt, -2, -1))
    tmp = (torch.einsum('...ii', tmp) - 1) / 2  # batched trace
    res = torch.acos(tmp.clamp(-1, 1))

    if degree is True:
        res = res * 180 / torch.pi

    return res


def relative_translation_error(delta_t, delta_t_gt):
    res = 2 * np.linalg.norm(delta_t - delta_t_gt) / (np.linalg.norm(delta_t) + np.linalg.norm(delta_t_gt))
    return res


def epipolar_error(flow, R, t, K):
    '''
      Compute algebra error based on epipolar constraint
    :param flow: (B, 2, H, W), 1->2
    :param R: (B, 3, 3), 1->2
    :param t: (B, 3, 1), 1->2
    :param K: (B, 3, 3)
    :return residual: (B, H, W), abs of epipolar error, always positive
    '''
    device = flow.device
    b, _, h, w = flow.shape
    v, u = torch.meshgrid(
        torch.arange(0, h, dtype=torch.float32, device=device),
        torch.arange(0, w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    v, u = v.view(h * w), u.view(h * w)
    uv_1 = torch.stack((u, v)).unsqueeze(0).repeat(b, 1, 1)  # (B, 2, H * W)

    flow_12 = flow.view(b, 2, -1)  # (B, 2, H * W)
    flow_12 = torch.stack((flow_12[:, 0], flow_12[:, 1]), dim=1)  # (B, 2, H * W)
    uv_2 = uv_1 + flow_12  # (B, 2, H * W)

    S = torch.tensor([
        [0, 0, 0], [0, 0, -1], [0, 1, 0],
        [0, 0, 1], [0, 0, 0], [-1, 0, 0],
        [0, -1, 0], [1, 0, 0], [0, 0, 0]
    ], dtype=torch.float32, device=device)
    S = S.unsqueeze(0).repeat(b, 1, 1)  # (B, 9, 3)
    skew_t = torch.matmul(S, t).reshape(b, 3, 3)  # (B, 3, 3)

    uv_1 = torch.stack((uv_1[:, 0], uv_1[:, 1], torch.ones_like(uv_1[:, 0])), dim=1)  # (B, 3, H * W)
    uv_1 = uv_1.transpose(1, 2).unsqueeze(-1)  # (B, H * W, 3, 1)
    uv_2 = torch.stack((uv_2[:, 0], uv_2[:, 1], torch.ones_like(uv_2[:, 0])), dim=1)  # (B, 3, H * W)
    uv_2 = uv_2.transpose(1, 2).unsqueeze(-1)  # (B, H * W, 3, 1)
    K_inverse = torch.inverse(K).unsqueeze(1).repeat(1, h * w, 1, 1)  # (B, H * W, 3, 3)
    K_inverse_T = K_inverse.transpose(-2, -1)  # (B, H * W, 3, 3)
    E_21 = torch.matmul(skew_t, R).unsqueeze(1).repeat(1, h * w, 1, 1)  # (B, H * W, 3, 3)

    residual = torch.matmul(
        torch.matmul(torch.matmul(torch.matmul(uv_2.transpose(-2, -1), K_inverse_T), E_21), K_inverse), uv_1)
    residual = residual.squeeze(-1).squeeze(-1).reshape(b, h, w)  # (B, H, W)
    residual = torch.abs(residual)

    return residual


def homography_error(flow, R, K):
    '''
      Compute algebra error based on homography
    :param flow: (B, 2, H, W), 1->2
    :param R: (B, 3, 3), 1->2
    :param K: (B, 3, 3)
    :return residual: (B, H, W), abs of homography error, always positive
    '''
    device = flow.device
    b, _, h, w = flow.shape
    v, u = torch.meshgrid(
        torch.arange(0, h, dtype=torch.float32, device=device),
        torch.arange(0, w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    v, u = v.view(h * w), u.view(h * w)
    uv_1 = torch.stack((u, v)).unsqueeze(0).repeat(b, 1, 1)  # (B, 2, H * W)

    flow_12 = flow.view(b, 2, -1)  # (B, 2, H * W)
    flow_12 = torch.stack((flow_12[:, 0], flow_12[:, 1]), dim=1)  # (B, 2, H * W)
    uv_2 = uv_1 + flow_12  # (B, 2, H * W)

    uv_1 = torch.stack((uv_1[:, 0], uv_1[:, 1], torch.ones_like(uv_1[:, 0])), dim=1)  # (B, 3, H * W)
    uv_1 = uv_1.transpose(1, 2).unsqueeze(-1)  # (B, H * W, 3, 1)
    uv_2 = torch.stack((uv_2[:, 0], uv_2[:, 1], torch.ones_like(uv_2[:, 0])), dim=1)  # (B, 3, H * W)
    uv_2 = uv_2.transpose(1, 2).unsqueeze(-1)  # (B, H * W, 3, 1)
    K_inverse = torch.inverse(K).unsqueeze(1).repeat(1, h * w, 1, 1)  # (B, H * W, 3, 3)
    K = K.unsqueeze(1).repeat(1, h * w, 1, 1)  # (B, H * W, 3, 3)
    R = R.unsqueeze(1).repeat(1, h * w, 1, 1)  # (B, H * W, 3, 3)

    residual1 = torch.norm(uv_2 - torch.matmul(K, torch.matmul(R, torch.matmul(K_inverse, uv_1))),
                           dim=2)  # (B, H * W, 1)
    residual2 = torch.norm(uv_1 - torch.matmul(K, torch.matmul(R.transpose(-2, -1), torch.matmul(K_inverse, uv_2))),
                           dim=2)  # (B, H * W, 1)
    residual = residual1 + residual2  # (B, H * W, 1)
    residual = residual.squeeze(-1).reshape(b, h, w)  # (B, H, W)
    residual = torch.abs(residual)

    return residual


def reprojection_error(flow, depth, R, t, K, pixel=False, return_mask=False):
    '''
      Compute reprojection error
    :param flow: (B, 2, H, W), 1->2
    :param depth: (B, H, W), 1
    :param R: (B, 3, 3), 1->2
    :param t: (B, 3, 1), 1->2
    :param K: (B, 3, 3)
    :param pixel: whether residual is on image plane
    :return residual: (B, H, W), abs of reprojection error, always positive
    '''
    b = flow.shape[0]
    device = flow.device
    height = flow.shape[-2]
    width = flow.shape[-1]

    v, u = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=device),
        torch.arange(0, width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    v, u = v.view(height * width), u.view(height * width)
    uv1_ref = torch.stack((u, v, torch.ones_like(u))).unsqueeze(0).repeat(b, 1, 1)  # (B, 3, H*W)
    xy1_ref = torch.matmul(torch.inverse(K), uv1_ref)  # (B, 3, H*W)

    # project to other views
    depth = depth.reshape(b, 1, -1)  # (B, 1, H*W)
    xyz_ref = depth * xy1_ref  # (B, 3, H*W)
    xyz_src = torch.matmul(R, xyz_ref) + t  # (B, 3, H*W)
    z_src = xyz_src[:, -1].clamp(min=1e-6).unsqueeze(1)  # 防止除以零, 也保证深度为正
    if return_mask:
        z_src_ = z_src.reshape(b, 1, height, width)
        valid_mask = torch.where(z_src_ < 0.01, 0.0, 1.0)  # 防止小深度干扰训练, 默认第一帧不会有小深度
    xy1_src = xyz_src / z_src  # (B, 3, H*W)
    uv1_src = torch.matmul(K, xy1_src)  # (B, 3, H*W)

    # compute residual
    if pixel:
        rigid_flow = (uv1_src[:, :2] - uv1_ref[:, :2]).reshape(b, 2, height, width)  # (B, 2, H, W)
        residual_flow = flow - rigid_flow  # (B, 2, H, W)
        residual = residual_flow.norm(dim=1)  # (B, H, W)
    else:
        rigid_flow = (xy1_src[:, :2] - xy1_ref[:, :2]).reshape(b, 2, height, width)  # (B, 2, H, W)
        flow = torch.matmul(torch.inverse(K[:, 0:2, 0:2]), flow.reshape(b, 2, -1)).reshape(b, 2, height,
                                                                                           width)  # (B, 2, H, W)
        residual_flow = flow - rigid_flow  # (B, 2, H, W)
        residual = residual_flow.norm(dim=1)  # (B, H, W)

    residual = residual.unsqueeze(1)

    if return_mask:
        return residual, valid_mask
    else:
        return residual


def depth_consistency_error(flow, depth1, depth2, R, t, K):
    '''
      Compute reprojection error
    :param flow: (B, 2, H, W), 1->2
    :param depth1: (B, H, W), 1
    :param depth2: (B, H, W), 1
    :param R: (B, 3, 3), 1->2
    :param t: (B, 3, 1), 1->2
    :param K: (B, 3, 3)
    :return residual: (B, H, W), abs of reprojection error, always positive
    '''
    b = flow.shape[0]
    device = flow.device
    height = flow.shape[-2]
    width = flow.shape[-1]

    v, u = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=device),
        torch.arange(0, width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    v, u = v.view(height * width), u.view(height * width)
    uv1_ref = torch.stack((u, v, torch.ones_like(u))).unsqueeze(0).repeat(b, 1, 1)  # (B, 3, H*W)
    xy1_ref = torch.matmul(torch.inverse(K), uv1_ref)  # (B, 3, H*W)

    # project to other views
    depth1 = depth1.reshape(b, 1, -1)  # (B, 1, H*W)
    xyz_ref = depth1 * xy1_ref  # (B, 3, H*W)
    xyz_src = torch.matmul(R, xyz_ref) + t  # (B, 3, H*W)
    z_src = xyz_src[:, -1].reshape(b, height, width)  # (B, H, W)

    residual = torch.abs(depth2 - z_src)  # (B, H, W)

    return residual


def compute_sceneflow(depth, R, t, K):
    '''
      Compute scene flow
    :param depth: (B, H, W), 1
    :param R: (B, 3, 3), 1->2
    :param t: (B, 3, 1), 1->2
    :param K: (B, 3, 3)
    :return sceneflow: (B, 3, H, W)
    '''
    b = depth.shape[0]
    device = depth.device
    height = depth.shape[-2]
    width = depth.shape[-1]

    v, u = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=device),
        torch.arange(0, width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    v, u = v.view(height * width), u.view(height * width)
    uv1_ref = torch.stack((u, v, torch.ones_like(u))).unsqueeze(0).repeat(b, 1, 1)  # (B, 3, H*W)
    xy1_ref = torch.matmul(torch.inverse(K), uv1_ref)  # (B, 3, H*W)

    # project to other views
    depth = depth.reshape(b, 1, -1)  # (B, 1, H*W)
    xyz_ref = depth * xy1_ref  # (B, 3, H*W)
    xyz_src = torch.matmul(R, xyz_ref) + t  # (B, 3, H*W)
    sceneflow = (xyz_src - xyz_ref).reshape(b, 3, height, width)  # (B, 3, H, W)

    return sceneflow


def save_cmap(x, path, cmap='magma', text=None, text_size=30, quiet=False):
    '''
      save heat map
    :param x: (H, W)
    :param path: saved path, e.g. path='/home/xiechen/image.jpg'
    :param cmap: 'magma', 'viridis', 'plasma', 'inferno', 'turbo'
    :param text: can put some text on image
    :param text_size:
    :param quiet: whether log on screen
    :return:
    '''
    try:
        x_np = x.cpu().numpy()
    except:
        x_np = x
    normalizer = mpl.colors.Normalize(vmin=x_np.min(), vmax=x_np.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    colormapped_im = (mapper.to_rgba(x_np)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    if text is not None:
        im = pil_image_add_text(im, text, left=0, top=0, text_color=(0, 255, 0), text_size=text_size)
    im.save(path)
    if not quiet:
        print(f'\033[1;32m [success] \033[0m', 'save cmap to {}'.format(path))


def pil_image_add_text(pil_img, text, left, top, text_color=(255, 0, 0), text_size=13):
    draw = ImageDraw.Draw(pil_img)
    fontStyle = ImageFont.truetype("NotoSerifCJK-Bold.ttc", text_size, encoding="utf-8")
    draw.text((left, top), text, text_color, font=fontStyle)
    return pil_img


class Config:
    def __init__(self, path=None, cfg=None):
        assert path is not None or cfg is not None

        if path is not None:
            with open(path, 'r') as f:
                cfg = yaml.safe_load(f.read())

        for (k, v) in cfg.items():
            setattr(self, k, Config(cfg=v) if isinstance(v, dict) else None if v == 'None' else v)


def read(file):
    if file.endswith('.float3'):
        return readFloat(file)
    elif file.endswith('.flo'):
        return readFlow(file)
    elif file.endswith('.ppm'):
        return readImage(file)
    elif file.endswith('.pgm'):
        return readImage(file)
    elif file.endswith('.png'):
        return readImage(file)
    elif file.endswith('.jpg'):
        return readImage(file)
    elif file.endswith('.pfm'):
        return readPFM(file)[0]
    else:
        raise Exception('don\'t know how to read %s' % file)


def write(file, data):
    if file.endswith('.float3'):
        return writeFloat(file, data)
    elif file.endswith('.flo'):
        return writeFlow(file, data)
    elif file.endswith('.ppm'):
        return writeImage(file, data)
    elif file.endswith('.pgm'):
        return writeImage(file, data)
    elif file.endswith('.png'):
        return writeImage(file, data)
    elif file.endswith('.jpg'):
        return writeImage(file, data)
    elif file.endswith('.pfm'):
        return writePFM(file, data)
    else:
        raise Exception('don\'t know how to write %s' % file)


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)


def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)[0]
        if len(data.shape) == 3:
            return data[:, :, 0:3]
        else:
            return data

    return misc.imread(name)


def writeImage(name, data):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return writePFM(name, data, 1)

    return misc.imsave(name, data)


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def readFloat(name):
    f = open(name, 'rb')

    if (f.readline().decode("utf-8")) != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def writeFloat(name, data):
    f = open(name, 'wb')

    dim = len(data.shape)
    if dim > 3:
        raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim == 2:
        data.tofile(f)

    else:
        np.transpose(data, (2, 0, 1)).tofile(f)


def percentage(l: np.array, th: float):
    return len(l[l > th]) / len(l)


def save_residual(residual, path='/home/xiechen/res.png', v_max=1000.0):
    '''

    :param residual: (H, W)
    :param path:
    :return:
    '''
    res = residual.cpu().numpy()
    normalizer = mpl.colors.Normalize(vmin=res.min(), vmax=min(res.max(), v_max))
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    res = (mapper.to_rgba(res)[:, :, :3] * 255).astype(np.uint8)
    res = res[:, :, [2, 1, 0]]
    res_pil = pil.fromarray(res)
    text = 'min: {:.2}\nmax: {:.2}'.format(residual.min(), residual.max())
    res_pil = pil_image_add_text(res_pil, text, left=0, top=0, text_color=(0, 255, 0), text_size=30)
    res = np.array(res_pil)
    return cv2.imwrite(path, res)


def show_residual(residual, v_max=1e9, image=None, mask=None):
    '''

    :param residual: (H, W)
    :param image: (C, H, W)
    :param mask: (1, H, W)
    :return:
    '''
    res = residual.cpu().numpy()
    normalizer = mpl.colors.Normalize(vmin=res.min(), vmax=min(res.max(), v_max))
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    res = (mapper.to_rgba(res)[:, :, :3] * 255).astype(np.uint8)
    res = res[:, :, [2, 1, 0]]
    res_pil = pil.fromarray(res)
    text = 'min: {:.4f}\nmax: {:.4f}'.format(residual.min(), residual.max())
    res_pil = pil_image_add_text(res_pil, text, left=0, top=0, text_color=(0, 255, 0), text_size=30)
    res = np.array(res_pil)

    if image is not None:
        if mask is not None:
            img = np.array(image.permute(1, 2, 0) * 255, dtype=np.uint8)
            mask = np.array(mask.permute(1, 2, 0).repeat(1, 1, 3) * 255, dtype=np.uint8)
            res = cv2.vconcat([img, res, mask])
        else:
            img = np.array(image.permute(1, 2, 0) * 255, dtype=np.uint8)
            res = cv2.vconcat([img, res])

    cv2.imshow('residual', res)
    cv2.waitKey(0)


def reproj_loss(pose, K, flow_l1l2, flow_l1r1=None, baseline=1.0, depth=None,
                mask=None, reduce=True, keepdim=False, mode='12'):
    """
        pose: T_12 (mode == '12'), T_21 (mode == '21'), in (rot, trans) format, which has 6 dimensions
        depth: (B, 1, H, W)
    """
    b = pose.shape[0]

    if depth is None:
        assert flow_l1r1 is not None
        fx = K[:, 0, 0]  # (B)
        disparity = torch.abs(flow_l1r1[:, 0]).clamp(min=1e-6)  # ensure not divided by zero when computing depth
        depth = baseline * fx[:, None, None] / disparity  # (B, H, W)
    else:
        depth = depth.squeeze(1)  # (B, H, W)

    if mode == '12':
        R_12 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_12 = pose[:, -3:][..., None]  # (B, 3, 1)
        R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
        t_21 = -torch.matmul(R_21, t_12)  # (B, 3, 1)
    elif mode == '21':
        R_21 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_21 = pose[:, -3:][..., None]  # (B, 3, 1)
    else:
        raise NotImplementedError('the mode of function reproj_loss can only choose from 12 and 21')

    residual = reprojection_error(flow_l1l2, depth, R_21, t_21, K).unsqueeze(1)  # (B, 1, H, W)

    if mask is None:
        mask = torch.ones_like(residual)

    if reduce and not keepdim:
        return (residual * mask).sum() / mask.sum()
    elif reduce and keepdim:
        return (residual * mask).view(b, -1).sum(dim=1) / mask.sum(dim=1)
    else:
        return residual


def reproj_loss_v2(pose, K, flow_l1l2, disp=None, baseline=1.0, depth=None,
                   mask=None, reduce=True, keepdim=False, mode='12'):
    """
        pose: T_12 (mode == '12'), T_21 (mode == '21'), in (rot, trans) format, which has 6 dimensions
        depth: (B, 1, H, W)
    """
    b = pose.shape[0]

    if depth is None:
        assert disp is not None
        assert len(disp.shape) == 4  # (B, 1, H, W)
        fx = K[:, 0, 0]  # (B)
        disparity = torch.abs(disp[:, 0]).clamp(min=1e-6)  # ensure not divided by zero when computing depth
        depth = baseline * fx[:, None, None] / disparity  # (B, H, W)
    else:
        depth = depth.squeeze(1)  # (B, H, W)

    if mode == '12':
        R_12 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_12 = pose[:, -3:][..., None]  # (B, 3, 1)
        R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
        t_21 = -torch.matmul(R_21, t_12)  # (B, 3, 1)
    elif mode == '21':
        R_21 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_21 = pose[:, -3:][..., None]  # (B, 3, 1)
    else:
        raise NotImplementedError('the mode of function reproj_loss can only choose from 12 and 21')

    residual, valid = reprojection_error(flow_l1l2, depth, R_21, t_21, K, return_mask=True)  # (B, 1, H, W)

    if mask is None:
        mask = torch.ones_like(residual)

    if reduce and not keepdim:
        return (residual * mask).sum() / mask.sum(), valid
    elif reduce and keepdim:
        return (residual * mask).view(b, -1).sum(dim=1) / mask.sum(dim=1), valid
    else:
        return residual, valid


def epipolar_distance(rot, trans, K, flow, mask=None, reduce=True):
    """
        compute epipolar error in epipolar distance
    :parma rot: (B, 3), R_12
    :param trans: (B, 3), t_12
    :param K: (B, 3, 3)
    :param flow: (B, 2, H, W), forward(1 -> 2)
    :param mask: (B, 1, H, W)
    :return sampson_dist: (B) / (B, 1, H, W)
    """
    trans = torch.nn.functional.normalize(trans, p=2, dim=-1)

    device = rot.device
    b, _, height, width = flow.shape
    v1, u1 = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=device),
        torch.arange(0, width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v1, u1 = v1.contiguous(), u1.contiguous()
    u1 = u1.unsqueeze(0).repeat(b, 1, 1)  # (B, H, W)
    v1 = v1.unsqueeze(0).repeat(b, 1, 1)  # (B, H, W)
    u2 = u1 + flow[:, 0]  # (B, H, W)
    v2 = v1 + flow[:, 1]  # (B, H, W)
    pts1 = torch.stack((u1, v1, torch.ones_like(u1)), dim=-1).reshape(b, -1, 3)  # (B, H*W, 3)
    pts2 = torch.stack((u2, v2, torch.ones_like(u2)), dim=-1).reshape(b, -1, 3)  # (B, H*W, 3)

    inv_K = torch.inverse(K)  # (B, 3, 3)
    R_12 = torch.matrix_exp(skew_matrix(rot))  # (B, 3, 3)
    t_12 = trans  # (B, 3)
    R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
    t_21 = (-R_21 @ t_12.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    skew_t_21 = skew_matrix(t_21)  # (B, 3, 3)
    F = inv_K.transpose(-2, -1) @ skew_t_21 @ R_21 @ inv_K  # (B, 3, 3)

    sampson_dist = kornia.geometry.epipolar.left_to_right_epipolar_distance(pts1, pts2, F)  # (B, H*W), kornia==0.6.12
    sampson_dist = sampson_dist.reshape(b, 1, height, width)  # (B, 1, H, W)

    if mask is not None:
        sampson_dist = sampson_dist * mask  # (B, 1, H, W)
    else:
        mask = torch.ones_like(sampson_dist, device=sampson_dist.device)

    if reduce:
        sampson_dist = torch.einsum('bchw->b', sampson_dist) / \
                       torch.maximum(torch.einsum('bchw->b', mask), torch.ones(b, device=device))  # (B)

    return sampson_dist


def sampson_distance(rot, trans, K, flow, mask=None, reduce=True):
    """
        compute epipolar error in Sampson distance
    :parma rot: (B, 3), R_12
    :param trans: (B, 3), t_12
    :param K: (B, 3, 3)
    :param flow: (B, 2, H, W), forward(1 -> 2)
    :param mask: (B, 1, H, W)
    :return sampson_dist: (B) / (B, 1, H, W)
    """
    trans = torch.nn.functional.normalize(trans, p=2, dim=-1)

    device = rot.device
    b, _, height, width = flow.shape
    v1, u1 = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32, device=device),
        torch.arange(0, width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    v1, u1 = v1.contiguous(), u1.contiguous()
    u1 = u1.unsqueeze(0).repeat(b, 1, 1)  # (B, H, W)
    v1 = v1.unsqueeze(0).repeat(b, 1, 1)  # (B, H, W)
    u2 = u1 + flow[:, 0]  # (B, H, W)
    v2 = v1 + flow[:, 1]  # (B, H, W)
    pts1 = torch.stack((u1, v1, torch.ones_like(u1)), dim=-1).reshape(b, -1, 3)  # (B, H*W, 3)
    pts2 = torch.stack((u2, v2, torch.ones_like(u2)), dim=-1).reshape(b, -1, 3)  # (B, H*W, 3)

    inv_K = torch.inverse(K)  # (B, 3, 3)
    R_12 = torch.matrix_exp(skew_matrix(rot))  # (B, 3, 3)
    t_12 = trans  # (B, 3)
    R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
    t_21 = (-R_21 @ t_12.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    skew_t_21 = skew_matrix(t_21)  # (B, 3, 3)
    F = inv_K.transpose(-2, -1) @ skew_t_21 @ R_21 @ inv_K  # (B, 3, 3)

    sampson_dist = kornia.geometry.epipolar.sampson_epipolar_distance(pts1, pts2, F)  # (B, H*W)
    sampson_dist = sampson_dist.reshape(b, 1, height, width)  # (B, 1, H, W)

    if mask is not None:
        sampson_dist = sampson_dist * mask  # (B, 1, H, W)
    else:
        mask = torch.ones_like(sampson_dist, device=sampson_dist.device)

    if reduce:
        sampson_dist = torch.einsum('bchw->b', sampson_dist) / \
                       torch.maximum(torch.einsum('bchw->b', mask), torch.ones(b, device=device))  # (B)

    return sampson_dist


def depth_consistency(pose, K, flow_l1l2,
                      disp1=None, disp2=None, baseline=1.0, depth1=None, depth2=None,
                      mask=None, reduce=True, keepdim=False, mode='12'):
    """
        pose: T_12 (mode == '12'), T_21 (mode == '21'), in (rot, trans) format, which has 6 dimensions
        depth: (B, 1, H, W)
    """
    b = pose.shape[0]

    if depth1 is None:
        assert disp1 is not None
        assert len(disp1.shape) == 4  # (B, 1, H, W)
        fx = K[:, 0, 0]  # (B)
        disp1 = torch.abs(disp1[:, 0]).clamp(min=1e-6)  # ensure not divided by zero when computing depth
        disp2 = torch.abs(disp2[:, 0]).clamp(min=1e-6)  # ensure not divided by zero when computing depth
        depth1 = baseline * fx[:, None, None] / disp1  # (B, H, W)
        depth2 = baseline * fx[:, None, None] / disp2  # (B, H, W)

    else:
        depth1 = depth1.squeeze(1)  # (B, H, W)
        depth2 = depth2.squeeze(1)  # (B, H, W)

    if mode == '12':
        R_12 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_12 = pose[:, -3:][..., None]  # (B, 3, 1)
        R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
        t_21 = -torch.matmul(R_21, t_12)  # (B, 3, 1)
    elif mode == '21':
        R_21 = so3_exp_map(pose[:, :3])  # (B, 3, 3)
        t_21 = pose[:, -3:][..., None]  # (B, 3, 1)
    else:
        raise NotImplementedError('the mode of function reproj_loss can only choose from 12 and 21')

    residual = depth_consistency_error(flow_l1l2, depth1, depth2, R_21, t_21, K).unsqueeze(1)  # (B, 1, H, W)

    if mask is None:
        mask = torch.ones_like(residual)

    if reduce and not keepdim:
        return (residual * mask).sum() / mask.sum()
    elif reduce and keepdim:
        return (residual * mask).view(b, -1).sum(dim=1) / mask.sum(dim=1)
    else:
        return residual


def argmin(lst):
    return min(range(len(lst)), key=lst.__getitem__)


def argmax(lst):
    return max(range(len(lst)), key=lst.__getitem__)


def skew(vec):
    """
        numpy version
    """
    x, y, z = vec[0], vec[1], vec[2]
    m = np.array([
        [0., -z, y],
        [z, 0., -x],
        [-y, x, 0.]
    ], dtype=np.float32)

    return m


def skew_matrix(vec):
    """
        return skew matrix from vector
    :param vec: (B, 3)
    :return skew_vec: (B, 3, 3)
    """
    b, dim = vec.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    skew_vec = torch.zeros((b, 3, 3), dtype=vec.dtype, device=vec.device)

    x, y, z = vec.unbind(1)

    skew_vec[:, 0, 1] = -z
    skew_vec[:, 0, 2] = y
    skew_vec[:, 1, 0] = z
    skew_vec[:, 1, 2] = -x
    skew_vec[:, 2, 0] = -y
    skew_vec[:, 2, 1] = x

    return skew_vec


def higgins_flow(intrinsic_layer, K, rot, trans=None, depth=None, normalize=True):
    """
        compute rotation flow and translation flow
        if not provide depth, translation flow will be normalized
    :param intrinsic_layer: (B, 2, H, W)
    :param K: (B, 3, 3)
    :parma rot: (B, 3), R_12
    :param trans: (B, 3), t_12
    :parm depth: None or (B, H, W)
    :param normalize: whether normalize trans_flow
    :return rot_flow, trans_flow: (B, 2, H, W)
    """
    b, _, height, width = intrinsic_layer.shape
    fx, fy = K[:, 0, 0], K[:, 1, 1]  # (B)
    f = torch.stack((fx, fy), dim=1)[:, :, None, None]  # (B, 2, 1, 1)
    xc, yc = intrinsic_layer[:, 0], intrinsic_layer[:, 1]  # (B, H, W)
    xc2, yc2, xcyc = xc * xc, yc * yc, xc * yc  # (B, H, W)

    # rotation flow
    # Higgins_theta -> R_12
    theta_x = rot[:, 0][:, None, None]  # (B, 1, 1)
    theta_y = rot[:, 1][:, None, None]  # (B, 1, 1)
    theta_z = rot[:, 2][:, None, None]  # (B, 1, 1)
    rot_flow_u = xcyc * theta_x - (1 + xc2) * theta_y + yc * theta_z  # (B, H, W)
    rot_flow_v = (1 + yc2) * theta_x - xcyc * theta_y - xc * theta_z  # (B, H, W)
    rot_flow = torch.stack((rot_flow_u, rot_flow_v), dim=1)  # (B, 2, H, W)
    rot_flow = rot_flow * f  # (B, 2, H, W)

    if trans is not None:
        # translation flow
        # Higgins_trans -> R_12^T @ t_12
        R_12 = torch.matrix_exp(skew_matrix(rot))  # (B, 3, 3)
        R_21 = R_12.transpose(-2, -1)  # (B, 3, 3)
        t_12 = trans.unsqueeze(-1)  # (B, 3, 1)
        t = (R_21 @ t_12).squeeze(-1)  # (B, 3)
        t_x = t[:, 0][:, None, None]  # (B, 1, 1)
        t_y = t[:, 1][:, None, None]  # (B, 1, 1)
        t_z = t[:, 2][:, None, None]  # (B, 1, 1)

        if depth is None:
            trans_flow_u = t_z * xc - t_x  # (B, H, W)
            trans_flow_v = t_z * yc - t_y  # (B, H, W)
            trans_flow = torch.stack((trans_flow_u, trans_flow_v), dim=1)  # (B, 2, H, W)
            if normalize:
                trans_flow = F.normalize(trans_flow, p=2, dim=1)  # (B, 2, H, W)
        else:
            trans_flow_u = (t_z * xc - t_x) / depth  # (B, H, W)
            trans_flow_v = (t_z * yc - t_y) / depth  # (B, H, W)
            trans_flow = torch.stack((trans_flow_u, trans_flow_v), dim=1)  # (B, 2, H, W)
            trans_flow = trans_flow * f

        return rot_flow, trans_flow
    else:
        return rot_flow


def save_network(network, path):
    saved_data = {'network': network.state_dict()}
    torch.save(saved_data, path)


def interpolate(pos, src, mode='bilinear', padding_mode='border'):
    """
        TODO: this function is used very frequently, so we need to be more efficient

        pos: (B, 2, H1, W1)
        src: (B, C, H2, W2)

        :returns (B, C, H1, W1)
    """
    height, width = src.shape[-2:]
    pos = pos.float()
    grid = torch.einsum('bchw->bhwc', pos)  # (B, H, W, 2)
    grid_x_normalized = grid[:, :, :, 0] / ((width - 1) / 2) - 1
    grid_y_normalized = grid[:, :, :, 1] / ((height - 1) / 2) - 1
    grid_normalized = torch.stack((grid_x_normalized, grid_y_normalized), dim=-1)  # (B, H, W, 2)
    sampled_src = F.grid_sample(src, grid_normalized, mode=mode, padding_mode=padding_mode)  # (B, C, H, W)

    return sampled_src


def init_uv(height, width, device='cpu'):
    v, u = torch.meshgrid(
        torch.arange(0, height, dtype=torch.float32),
        torch.arange(0, width, dtype=torch.float32),
        indexing='ij'
    )
    v, u = v.contiguous(), u.contiguous()
    uv = torch.stack((u, v)).to(device).unsqueeze(0)  # (B, 2, H, W)

    return uv


def pose_inv(pose: torch.Tensor):
    # device = pose.device
    # pose_np = pose.squeeze(0).cpu().numpy()  # (6)
    # T_12 = se3_to_matrix(pose_np)  # (4, 4)
    # T_21 = np.linalg.inv(T_12)
    # pose_inv_np = matrix_to_se3(T_21)
    # pose_inv = torch.tensor(pose_inv_np).unsqueeze(0).to(device)
    pose_inv = -pose

    return pose_inv


def flow_consistency_mask(uv1, flow_l1l2, flow_l2l1, th=1.0, quick=False):
    """
        uv1: (B, 2, H, W), float
        flow_l1l2: (B, 2, H, W)
        flow_l2l1: (B, 2, H, W)
        th: threshold of forward and backward consistency [px]
        quick: if True, means uv1 is not float coordinate

        M_flow: (B, 1, H, W)
    """
    if quick:
        sampled_flow_l1l2 = flow_l1l2
    else:
        sampled_flow_l1l2 = interpolate(uv1, flow_l1l2)
    uv2 = uv1 + sampled_flow_l1l2  # (B, 2, H, W)
    sampled_flow_l2l1 = interpolate(uv2, flow_l2l1)

    diff = sampled_flow_l1l2 + sampled_flow_l2l1  # (B, 2, H, W)
    M_flow = torch.where(torch.norm(diff, dim=1, keepdim=True) > th, 0.0, 1.0)  # (B, 1, H, W)

    return M_flow


def disp_to_depth(flow_l1r1, f, baseline):
    """
        f: (B)
        baseline: (B, 1)
        flow_l1r1: (B, 2, H, W), RAFT result of left with right
    """
    disparity = torch.abs(flow_l1r1[:, 0]).clamp(min=1e-6)  # ensure not divided by zero when computing depth
    depth = baseline * f[:, None, None] / disparity  # (B, H, W)
    depth = depth.unsqueeze(1)  # (B, 1, H, W)
    return depth


def disp_to_depth_v2(disp, f, baseline):
    """
        disp: (B, 1, H, W)
        f: (B)
        baseline: (B, 1)
    """
    disparity = torch.abs(disp).clamp(min=1e-6)  # ensure not divided by zero when computing depth
    depth = baseline * f[:, None, None, None] / disparity  # (B, 1, H, W)
    return depth


def depth_mask(depth, d_min=10.0, d_max=30.0):
    """
        depth: (B, 1, H, W)
    """
    cond = (depth < d_min) + (depth > d_max)
    M_depth = torch.where(cond, 0, 1)
    return M_depth


def avg_norm(src, mask):
    """
        src: (B, C, H, W)
        mask: (B, 1, H, W)

        NOTE: will return np.nan if mask.sum()==0
    """
    return torch.norm(src * mask, dim=1, keepdim=True).sum() / mask.sum()


def quat_to_matrix(pose):
    """
        pose[:3]   -> translation (tx, ty, tz), np.array
        pose[-4:]  -> rotation (qx, qy, qz, qw), np.array
    """
    assert len(pose) == 7, 'unmatched shape: {} with (7)'.format(pose.shape)

    r = R2.from_quat(pose[-4:])
    R_12 = r.as_matrix()
    t_12 = np.array(pose[:3])
    T_12 = np.eye(4)
    T_12[0:3, 0:3] = R_12
    T_12[0:3, 3] = t_12

    return T_12


def se3_to_matrix(pose):
    """
        pose[:3]   -> rotation (in so3), np.array
        pose[-3:]  -> translation, np.array
    """
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    assert len(pose) == 6, 'unmatched shape: {} with (6)'.format(pose.shape)

    r = R2.from_rotvec(pose[:3])
    R_12 = r.as_matrix()
    t_12 = np.array(pose[-3:])
    T_12 = np.eye(4)
    T_12[0:3, 0:3] = R_12
    T_12[0:3, 3] = t_12

    return T_12


def matrix_to_se3(T):
    """
        T   -> transform (in SE3), np.array
        pose  -> translation (in se3), np.array
    """
    has_batch = len(T.shape) == 3
    if has_batch:
        r = R2.from_matrix(T[:, :3, :3])
        rot = r.as_rotvec()
        trans = T[:, :3, 3]
        pose = np.hstack((rot, trans))
    else:
        r = R2.from_matrix(T[:3, :3])
        rot = r.as_rotvec()
        trans = T[:3, 3]
        pose = np.hstack((rot, trans))

    return pose


def select_by_mask(src_list, mask, N_select):
    """
        select N samples from source, considered mask

        src_list: [src], src -> (B, C, H, W), B=1
        mask: (B, 1, H, W), 1 for valid
    """
    assert mask.shape[0] == 1  # not yet implement for multi-batch
    N_valid = int(mask.sum().item())
    select_id = torch.randperm(N_valid)[:N_select]  # random seed do not have effects ???

    select_src_list = []
    for src in src_list:
        assert src.shape[-2:] == mask.shape[-2:]
        c = src.shape[1]
        valid_src = torch.masked_select(src, mask.bool()).reshape(c, -1)  # (C, N)
        valid_src = valid_src.transpose(0, 1).contiguous()  # (N, C)
        select_src = valid_src[select_id]
        select_src = select_src.cpu().numpy()  # (N, C)
        select_src_list.append(select_src)

    return select_src_list


def proj(p, K):
    """
        project N points to pixel plane

        p: (N, 3), in camera coordinate
        K: (3, 3)

        :return uv -> (N, 2)
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = p[:, 0] / p[:, 2]
    y = p[:, 1] / p[:, 2]
    uv = np.stack((fx * x + cx, fy * y + cy), axis=-1)

    return uv


def unproj(uv, K, depth):
    """
        unproject N points

        uv: (N, 2)
        K: (3, 3)
        depth: (N, 1)

        :return point_3d -> (N, 3)
    """
    N = uv.shape[0]
    u, v = uv[:, 0], uv[:, 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    uv_n = np.stack(((u - cx) / fx, (v - cy) / fy, np.repeat(1, N)), axis=-1)  # (N, 3)
    point_3d = uv_n * depth  # (N, 3)

    return point_3d


def transform(p, T):
    """
        p: (N, 3), 1
        T: (4, 4), 1 -> 2

        :return  (N, 3)
    """
    assert p.shape[-1] == 3
    N = p.shape[0]
    p_ = np.concatenate((p, np.expand_dims(np.repeat(1, N), -1)), axis=-1)  # (N, 4)
    q = np.matmul(T[None, :, :], p_[:, :, None])

    return q[:, :3, 0]


class Timer(ContextDecorator):
    def __init__(self, name, enabled=True, N=1):
        self.name = name
        self.enabled = enabled
        self.N = N

        if self.enabled:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()

    def __exit__(self, type, value, traceback):
        # global all_times
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()

            elapsed = self.start.elapsed_time(self.end) / self.N
            # all_times.append(elapsed)
            print(f"{self.name}:  {elapsed:.03f} ms")


def nan(n: int) -> np.ndarray:
    v = np.empty(n)
    v[:] = np.nan
    return v


def bound(x, v_min, v_max):
    return max(min(x, v_max), v_min)


def is_in_image(uv: np.ndarray, height: int, width: int) -> bool:
    u, v = uv[0], uv[1]

    if u < 0 or u >= width:
        return False

    if v < 0 or v >= height:
        return False

    return True


def draw_1d(x, y, xlabel='x', ylabel='y', interpolate=False):
    n = len(x)
    assert n == len(y)

    if interpolate:
        n_new = 10 * n
        f = inter.interp1d(x, y, kind='quadratic')
        x_new = np.linspace(min(x), max(x), n_new)
        y_new = f(x_new)
    else:
        x_new = x
        y_new = y

    # plot
    fig, ax = plt.subplots()
    ax.set_yscale("log", base=10)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.plot(x_new, y_new, linewidth=2.0)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()


def to_vehicle_2D_pattern(T_wc: np.ndarray):
    assert T_wc.shape == (4, 4)

    rot_vec = R2.from_matrix(T_wc[:3, :3]).as_rotvec()
    rot_vec[0] = 0.0  # pitch
    rot_vec[2] = 0.0  # roll
    T_wc[:3, :3] = R2.from_rotvec(rot_vec).as_matrix()

    T_wc[1, -1] = 0.0  # y axis

    return T_wc


def adjust_intrinsic(K: np.ndarray, raw_size: tuple, mode: str,
                     to_left=None, to_top=None, scale=None) -> np.ndarray:
    """
        K: intrinsic matrix -> 3 x 3
        raw_size: raw image size -> (h, w)
    """
    assert K.shape == (3, 3)
    raw_h, raw_w = raw_size
    raw_fx, raw_fy = K[0, 0], K[1, 1]
    raw_cx, raw_cy = K[0, 2], K[1, 2]

    if mode == 'rotate':
        fx = raw_fy
        fy = raw_fx
        cx = raw_cy
        cy = raw_w - raw_cx

        new_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    elif mode == 'crop':
        assert to_left is not None
        assert to_top is not None

        fx = raw_fx
        fy = raw_fy
        cx = raw_cx - to_left
        cy = raw_cy - to_top

        new_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    elif mode == 'scale':
        assert scale is not None

        if isinstance(scale, tuple):
            scale_x, scale_y = scale[0], scale[1]
        elif isinstance(scale, float):
            scale_x, scale_y = scale, scale
        else:
            raise NotImplementedError

        fx = raw_fx * scale_x
        fy = raw_fy * scale_y
        cx = raw_cx * scale_x
        cy = raw_cy * scale_y

        new_K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    else:
        new_K = K
        print('No adjustment applied to intrinsic.')

    return new_K


def epipolar_score(p1, p2, T_21, K, normalize=True, scale=1):
    R_21 = T_21[:3, :3]
    t_21 = T_21[:3, 3]
    if normalize:
        t_21 = t_21 / np.linalg.norm(t_21)
    E = skew(t_21) @ R_21

    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    p1 = np.hstack((p1, 1))
    p2 = np.hstack((p2, 1))

    s = p2.T @ F @ p1

    return abs(s) * scale


def freeze(layers_to_freeze: list):
    for layer in layers_to_freeze:
        for param in layer.parameters():
            param.required_grad = False


def unfreeze(layers_to_unfreeze: list):
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.required_grad = True


def gaussian_blur(img, kernel_size=3, device='cuda'):
    """
        img: (B, C, H, W)
    """
    in_channels = img.shape[1]
    assert in_channels == 3  # 3 channels
    out_channels = in_channels
    groups = in_channels

    kernel_size2 = kernel_size ** 2
    smooth_kernel = 1.0 / kernel_size2 * torch.ones((out_channels, in_channels // groups, kernel_size, kernel_size),
                                                    dtype=torch.float32,
                                                    device=device)  # (B, C, 3, 3)
    smooth_img = F.conv2d(img, smooth_kernel, padding=(kernel_size - 1) // 2, groups=groups)  # (B, C, 3, 3)

    return smooth_img


def patchify(img, window_size=(8, 8)):
    """
        img: (B, C, H, W)
    """
    b = img.shape[0]
    win_h, win_w = window_size
    out_img = rearrange(img, 'b c (N_h win_h) (N_w win_w) -> (b N_h N_w) c win_h win_w', win_h=win_h, win_w=win_w)
    return out_img


def unfold_patchify(img, window_size=(11, 11)):
    """
        img: (B, C, H, W)

        return: out_img: (B * H * W, C , window_H, window_W)
    """
    b, c, h, w = img.shape
    win_h, win_w = window_size
    out_img = F.unfold(img, kernel_size=window_size, padding=(win_h//2, win_w//2))
    out_img = out_img.reshape(b, c, win_h, win_w, h, w)
    out_img = rearrange(out_img, 'b c win_h win_w N_h N_w -> (b N_h N_w) c win_h win_w', win_h=win_h, win_w=win_w)

    return out_img


def warp_image(img, flow):
    device = img.device
    b, _, h, w = img.shape
    uv1 = init_uv(h, w, device)
    uv2 = uv1 + flow
    warped_img = interpolate(uv2, img)
    return warped_img


def compute_dense_ssim(img1, flow, kernel_size=11, gaussian_kernel=5):
    b, _, h, w = img1.shape
    img2 = warp_image(img1, flow)
    patched_left_img1 = unfold_patchify(img1, window_size=(kernel_size, kernel_size))
    patched_left_img2 = unfold_patchify(img2, window_size=(kernel_size, kernel_size))
    ssim = kornia.metrics.ssim(patched_left_img1, patched_left_img2,
                               gaussian_kernel, max_val=1.0, eps=1e-12, padding='same')
    ssim = ssim.mean(dim=(-2, -1))
    ssim = rearrange(ssim, '(b h w) c -> b c h w', h=h, w=w)
    ssim_loss = (1 - ssim) / 2
    ssim_loss = ssim_loss.mean(dim=1, keepdim=True)

    return ssim_loss


def interpolate_pose(tgt_times, src_times, src_poses):
    # 超过范围的都舍掉: [start, end)
    start = np.argmax(src_times[0] < tgt_times)
    end = np.argmax(src_times[-1] < tgt_times)
    if end == 0:
        end = len(tgt_times)
    tgt_times = tgt_times[start: end]

    # 平移插值
    src_translations = src_poses[:, :3, 3]
    interp_trans = interp1d(src_times, src_translations, axis=0, kind='linear', fill_value="extrapolate")
    tgt_trans = interp_trans(tgt_times)

    # 旋转插值（球面插值）
    src_rotations = src_poses[:, :3, :3]
    src_rots = R2.from_matrix(src_rotations)
    slerp = Slerp(src_times, src_rots)
    tgt_rots = slerp(tgt_times)
    tgt_rotations = tgt_rots.as_matrix()

    # 构造齐次变换矩阵 [R|t]
    tgt_poses = []
    for t, R in zip(tgt_trans, tgt_rotations):
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t
        tgt_poses.append(pose)

    return np.array(tgt_poses), start, end


























