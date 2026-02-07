'''
  Modified on vit-pytorch, add mask attention
  Xie Chen, 2023.12.24
'''

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

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


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=kernel_size//2,
                              stride=stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# classes
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, ln_layer=False):
        super().__init__()
        if ln_layer:
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, ln_layer=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        if ln_layer:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

        self.softmax = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # TODO (*): add mask
        if mask is not None:
            b = dots.shape[0]
            mask_ = mask.reshape(b, 1, -1, 1)
            dots_mask = torch.matmul(mask_, mask_.transpose(-2, -1))
            dots_mask = (1 - dots_mask).bool()  # 0 is masked to 1 is masked
            dots.masked_fill_(dots_mask, -1e9)
            attn = self.softmax(dots)

            # some rows may be all masked, so they do average indeed, replace them with zero
            attn = attn.masked_fill(dots_mask, 0.0)

        else:
            attn = self.softmax(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, ln_layer=True):
        super().__init__()
        if ln_layer:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, ln_layer=ln_layer),
                FeedForward(dim, mlp_dim, ln_layer=ln_layer)
            ]))

    def forward(self, x, mask=None):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x, mask) + x
            x = ff(x) + x

            # TODO (*): optional
            if mask is not None:
                b = mask.shape[0]
                mask_ = mask.reshape(b, 1, -1).permute(0, 2, 1)
                x = x * mask_

        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64, mask=None, ln_layer=False):
        super().__init__()
        self.patch_size = patch_size
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        if ln_layer:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, ln_layer=ln_layer)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img, mask=None, feature=False):
        device = img.device
        b, c, h, w = img.shape

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # TODO (*): optional
        if mask is not None:
            b = mask.shape[0]
            mask_ = mask.reshape(b, 1, -1).permute(0, 2, 1)
            x = x * mask_

        x = self.transformer(x, mask)  # (B, H*W, 256)

        # TODO: local mask pooling layer
        if not feature:
            # if mask is not None:
            #     b = mask.shape[0]
            #     mask = mask.reshape(b, 1, -1).transpose(1, 2)  # (B, H*W, 1)
            #     num_mask = mask.sum(dim=1).clamp(min=1e-4)  # (B, 1), if mask all image, should avoid dividing by zero
            #     x = (mask * x).sum(dim=1) / num_mask  # (B, 256)
            # else:
            #     x = x.mean(dim=1)  # (B, 256)

            x = x.mean(dim=1)  # (B, 256)

            return self.linear_head(x)

        else:
            x = self.linear_head(x)
            x = x.reshape(b, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2)

            return x
