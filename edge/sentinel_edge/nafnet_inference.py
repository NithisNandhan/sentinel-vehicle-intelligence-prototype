"""Inference-only NAFNet width32, adapted from megvii-research/NAFNet.

Copyright (c) 2022 megvii-model. MIT licensed; see third_party/NAFNET_LICENSE.txt.
Source revision 2b4af71ebe098a92a75910c233a3965a3e93ede4.
LayerNorm uses equivalent PyTorch operations instead of custom autograd.
No training, BasicSR dependency or checkpoint deserialisation lives here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        centered = x - x.mean(1, keepdim=True)
        normalized = centered / (centered.square().mean(1, keepdim=True) + 1e-6).sqrt()
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, channels * 2, 1)
        self.conv5 = nn.Conv2d(channels, channels, 1)
        self.norm1, self.norm2 = LayerNorm2d(channels), LayerNorm2d(channels)
        self.dropout1, self.dropout2 = nn.Identity(), nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.sg(self.conv2(self.conv1(self.norm1(x))))
        y = x + self.conv3(y * self.sca(y)) * self.beta
        return y + self.conv5(self.sg(self.conv4(self.norm2(y)))) * self.gamma


class NAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.intro = nn.Conv2d(3, 32, 3, padding=1)
        self.ending = nn.Conv2d(32, 3, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        channels = 32
        for blocks in (1, 1, 1, 28):
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(blocks))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
            channels *= 2
        self.middle_blks = nn.Sequential(NAFBlock(channels))
        for _ in range(4):
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(NAFBlock(channels)))

    def forward(self, image):
        h, w = image.shape[-2:]
        if max(h, w) > 256:
            raise ValueError('NAFNet inference bound exceeded')
        padded = F.pad(image, (0, (-w) % 16, 0, (-h) % 16))
        x, skips = self.intro(padded), []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        return (self.ending(x) + padded)[:, :, :h, :w]
