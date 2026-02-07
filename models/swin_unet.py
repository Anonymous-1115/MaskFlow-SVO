import sys
import time
import math
from thop import profile, clever_format

import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat
from torchvision.ops.misc import Conv2dNormActivation
import torch.nn.functional as F
from einops import rearrange

from models.simple_vit import SimpleViT
from utils import *


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x):
        return self.fc(x)


def masked_avg_pool(input_tensor, kernel_size, mask=None):
    b, c, h, w = input_tensor.shape

    if isinstance(kernel_size, int):
        patch_h, patch_w = kernel_size, kernel_size
    elif isinstance(kernel_size, tuple):
        patch_h, patch_w = kernel_size
    else:
        raise NotImplementedError

    # padding if necessary (zero padding)
    if h % patch_h != 0 or w % patch_w != 0:
        new_h = math.ceil(h / patch_h) * patch_h
        new_w = math.ceil(w / patch_w) * patch_w
        pad_h = (new_h - h) // 2
        pad_w = (new_w - w) // 2
        input_tensor = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h))

        if mask is not None:
            mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h))
            assert input_tensor.shape[-2:] == mask.shape[-2:]

    input_tensor = rearrange(input_tensor, 'b c (h p1) (w p2) -> b c h w (p1 p2)', p1=patch_h, p2=patch_w)

    if mask is not None:
        mask = rearrange(mask, 'b c (h p1) (w p2) -> b c h w (p1 p2)', p1=patch_h, p2=patch_w)
        masked_input = input_tensor * mask
        avg = masked_input.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-4)

    else:
        avg = input_tensor.mean(dim=-1)

    return avg


class ViTDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, patch_size, ln_layer=False):
        super().__init__()
        assert out_channels == 6

        self.vision_transformer = SimpleViT(
            channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            num_classes=128,
            dim=128,
            depth=3,  # <- TODO
            heads=8,  # <- TODO
            dim_head=32,
            mlp_dim=128,
            ln_layer=ln_layer
        )

        bias = True
        self.rot_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 6, 128, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 32, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 3, bias=bias)
        )

        self.trans_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 6, 128, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 32, bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 3, bias=bias)
        )

    def forward(self, x, mask=None):
        """
            x: (B, C, H, W)
        """
        latent = self.vision_transformer(x, mask=mask, feature=True)  # 12 x 40 -> 12 x 40

        # TODO (*): optional
        if mask is not None:
            latent = latent * mask

        latent = masked_avg_pool(latent, kernel_size=(7, 7), mask=mask)  # 12 x 40 -> 2 * 6

        rot = self.rot_head(latent)
        trans = self.trans_head(latent)

        pose = torch.cat([rot, trans], dim=-1)

        return pose


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, ln_layer):
        super().__init__()
        if ln_layer:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        # return self.norm(self.fn(x, **kwargs))   # TODO: V2


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        self.rearrange_mask = lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) d (w_h w_w)',
                                                  w_h=self.window_size, w_w=self.window_size, d=1)

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask=None):
        if self.shifted:
            x = self.cyclic_shift(x)

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        # print('({}, {}) -> ({}, {})x{}'.format(n_h, n_w, nw_h, nw_w, self.window_size))

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        # TODO: add mask
        if mask is not None:
            mask = self.rearrange_mask(mask)  # (B, H, W, 1) -> (B, 1, 块数, 1， 子块数)
            mask = mask.repeat(1, self.heads, 1, self.window_size * self.window_size, 1)  # (B, heads, 块数, 子块数， 子块数)
            mask = (1 - mask).bool()  # 0 is masked to 1 is masked
            dots.masked_fill_(mask, -1e9)

        # dots: (B, heads, 块数, 子块数， 子块数)
        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, ln_layer=True):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding),
                                                ln_layer=ln_layer))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim),
                                          ln_layer=ln_layer))

    def forward(self, x, mask=None):
        x = self.attention_block(x, mask=mask)
        x = self.mlp_block(x)
        return x


class PatchMerging_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=downscaling_factor,
            stride=downscaling_factor,
            padding=0
        )

    def forward(self, x):
        x = self.patch_merge(x).permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, in_channels, out_channels, upscaling_factor, bilinear=False):
        super().__init__()
        self.upscaling_factor = upscaling_factor
        self.bilinear = bilinear

        if self.bilinear:
            self.upsample = nn.Upsample(scale_factor=upscaling_factor, mode='bilinear', align_corners=False)
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False)
        else:
            self.linear = nn.Linear(in_channels, out_channels * upscaling_factor ** 2, bias=False)
            self.patch_expand = lambda t: rearrange(t, 'b h w (p1 p2 c)-> b c (h p1) (w p2)',
                                                    p1=upscaling_factor,
                                                    p2=upscaling_factor,
                                                    c=out_channels)
            self.conv = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  bias=False)

    def forward(self, x):
        if self.bilinear:
            x = self.upsample(x)
            x = self.conv(x)  # (B, C, H, W)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.linear(x)
            x = self.patch_expand(x)  # (B, C, H, W)
            x = self.conv(x)  # (B, C, H, W)

        return x


class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, ln_layer=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        # TODO: 这块如何和 mask 机制结合？
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          ln_layer=ln_layer),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          ln_layer=ln_layer),
            ]))

    def forward(self, x, mask=None):
        x = self.patch_partition(x)  # downscale
        for regular_block, shifted_block in self.layers:
            x = regular_block(x, mask=mask)
            x = shifted_block(x, mask=mask)
        return x.permute(0, 3, 1, 2)


class StageModule_Up(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, upscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, ln_layer=True):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_expand = PatchExpand(in_channels=in_channels, out_channels=hidden_dimension,
                                        upscaling_factor=upscaling_factor)
        self.concat_linear = nn.Linear(2 * hidden_dimension, hidden_dimension)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          ln_layer=ln_layer),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          ln_layer=ln_layer),
            ]))

    def forward(self, x, x_inner):
        """
            x: (B, C, H, W)
            x_inner: (B, C, H, W)

            return: (B, C, H, W)
        """
        x = self.patch_expand(x).permute(0, 2, 3, 1)  # upscale, (B, H, W, C)
        x_inner = x_inner.permute(0, 2, 3, 1)
        x = torch.cat([x, x_inner], dim=-1)
        x = self.concat_linear(x)  # (B, H, W, C)

        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


class SwinPoseNet(nn.Module):
    def __init__(self, *, input_size, hidden_dim, layers, heads, channels=3, head_dim=32,
                 window_size=(7, 7, 7, 7), downscaling_factors=(4, 2, 2, 2), th_conf=0.5,
                 relative_pos_embedding=True, multi_scale=False, ln_layer=True):
        super().__init__()
        self.height, self.width = input_size
        self.vit_factor = int(np.prod(np.array(downscaling_factors)))
        self.vit_height = self.height // self.vit_factor
        self.vit_width = self.width // self.vit_factor
        self.th_conf = th_conf

        self.stages_down = nn.ModuleList([])
        self.num_stages = len(window_size)
        self.downscaling_factors = downscaling_factors

        self.stages_up = nn.ModuleList([])

        self.inner_blocks = nn.ModuleList([])
        self.multi_scale = multi_scale

        # encoder
        for i in range(self.num_stages):
            if i == 0:
                stage = StageModule(in_channels=channels,
                                    hidden_dimension=hidden_dim,
                                    layers=layers[i],
                                    downscaling_factor=downscaling_factors[i],
                                    num_heads=heads[i],
                                    head_dim=head_dim,
                                    window_size=window_size[i],
                                    relative_pos_embedding=relative_pos_embedding,
                                    ln_layer=ln_layer)
                self.stages_down.append(stage)
            else:
                stage = StageModule(in_channels=hidden_dim * 2 ** (i - 1),
                                    hidden_dimension=hidden_dim * 2 ** i,
                                    layers=layers[i],
                                    downscaling_factor=downscaling_factors[i],
                                    num_heads=heads[i],
                                    head_dim=head_dim,
                                    window_size=window_size[i],
                                    relative_pos_embedding=relative_pos_embedding,
                                    ln_layer=ln_layer)
                self.stages_down.append(stage)
                self.inner_blocks.append(
                    Conv2dNormActivation(
                        in_channels=hidden_dim * 2 ** (i - 1),
                        out_channels=hidden_dim * 2 ** (self.num_stages - 1),
                        kernel_size=1,
                        padding=0,
                        norm_layer=None,
                        activation_layer=None
                    )
                )

        # pose decoder
        self.vit_decoder = ViTDecoder(
            in_channels=hidden_dim * 2 ** (self.num_stages - 1),
            out_channels=6,
            image_size=(self.vit_height, self.vit_width),
            patch_size=1,
            ln_layer=ln_layer
        )

        # confidence decoder
        for i in range(self.num_stages):
            if i == 0:
                stage = nn.Identity()
                self.stages_up.append(stage)
            else:
                stage = StageModule_Up(in_channels=hidden_dim * 2 ** (self.num_stages - i),
                                       hidden_dimension=hidden_dim * 2 ** (self.num_stages - i - 1),
                                       layers=layers[-i - 1],
                                       upscaling_factor=downscaling_factors[-i],
                                       num_heads=heads[-i - 1],
                                       head_dim=head_dim,
                                       window_size=window_size[-i - 1],
                                       relative_pos_embedding=relative_pos_embedding,
                                       ln_layer=ln_layer)
                self.stages_up.append(stage)

        self.final_expand = PatchExpand(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            upscaling_factor=downscaling_factors[0],
            bilinear=True
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, prior_masks=None, flow_mask=None):
        b, _, h, w = x.shape
        features = []  # -> confidence
        for stage in self.stages_down:
            x = stage(x)  # (B, C, H, W)
            features.append(x)

        for i, stage in enumerate(self.stages_up):
            if i == 0:
                continue
            x = stage(x, features[-i - 1])  # (B, C, H, W)

        x = self.final_expand(x)  # (B, C, H, W)
        conf = self.conf_head(x)  # (B, 1, H, W), [0, 1]

        final_feature = features[-1]  # -> pose
        if self.multi_scale:
            for i, f in enumerate(features):
                if i == len(features) - 1:
                    break
                f_ = self.inner_blocks[i](
                    F.interpolate(f,
                                  scale_factor=1 / 2 ** (self.num_stages - 1 - i),
                                  mode="nearest")
                )
                final_feature = final_feature + f_

        # generate dynamic mask
        mask = torch.where(conf < self.th_conf, 0.0, 1.0)
        if flow_mask is not None:
            mask = mask * flow_mask

        # neural_mask = mask.clone()

        with torch.no_grad():
            if prior_masks is not None:
                neural_mask_ = mask.bool().logical_not()  # 取反
                act = (torch.logical_and(neural_mask_, prior_masks)).any(dim=(-2, -1))[:, :, None, None]
                act_prior_mask = torch.logical_and(act, prior_masks).any(dim=1, keepdim=True)
                mask = torch.logical_or(neural_mask_, act_prior_mask)
                mask = mask.logical_not().float()  # 取反

                # cv2_show_depth(neural_mask.cpu(), shape='bchw')
                # cv2_show_depth(torch.prod(1 - prior_masks.float(), dim=1, keepdim=True).cpu(), shape='bchw')
                # cv2_show_depth(mask.cpu(), shape='bchw')

        mask_vit = F.interpolate(mask, scale_factor=1 / self.vit_factor, mode="nearest")

        pose = self.vit_decoder(final_feature, mask=mask_vit)  # (B, 6)

        return pose, conf, mask


if __name__ == '__main__':
    from tqdm import tqdm

    c = 5
    net = SwinPoseNet(
        input_size=(192, 640),
        hidden_dim=32,
        layers=(2, 2, 2),
        heads=(3, 6, 9),
        channels=c,
        head_dim=32,
        window_size=(8, 8, 4),  # window_size 要整除下采样后的尺寸
        downscaling_factors=(4, 2, 2),  # (192, 640) | /4-> (48, 160) /2-> (24, 80) /2-> (12, 40)
        relative_pos_embedding=True,
        multi_scale=True,
        th_conf=0.5,
        ln_layer=False
    )
    net = net.cuda().eval()

    x = torch.randn(1, c, 192, 640).cuda()
    print('input shape: {}'.format(x.shape))

    y, conf, mask = net(x)
    print('output shape: {}, {}'.format(y.shape, conf.shape))

    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], '%.2f')
    print(f'FLOPs: {flops}, Params: {params}')

    for _ in range(100):
        y, conf, mask = net(x)

    start = time.perf_counter_ns()

    N = 10000
    for _ in tqdm(range(N)):
        y, conf, mask = net(x)

    end = time.perf_counter_ns()
    time_per_run = (end - start) * 1e-6 / N
    print('Time: {} ms'.format(time_per_run))
