import contextlib
import logging
import os
import types

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F



class FeatureFlat(nn.Module):
    """
    a few conv layers to flatten features that come out of ResNet
    """

    def __init__(self, input_shape, output_size):
        super().__init__()
        if input_shape[0] == -1:
            input_shape = input_shape[1:]
        layers, activation_shape = self.init_cnn(input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0])
        layers += [Flatten(), nn.Linear(np.prod(activation_shape), output_size)]
        self.layers = nn.Sequential(*layers)

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(
                    planes_in,
                    planes_out,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(planes_out),
                nn.ReLU(inplace=True),
            ]
            planes_in = planes_out
            spatial = (spatial - kernel + 2 * padding) // stride + 1
        activation_shape = (planes_in, spatial, spatial)
        return layers, activation_shape

    def forward(self, frames):
        activation = self.layers(frames)
        return activation


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
