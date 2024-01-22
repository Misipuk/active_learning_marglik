import torch
from torch import nn
import numpy as np

from asdfghjkl.operations import Bias, Scale


ACTIVATIONS = ['relu', 'selu', 'tanh', 'silu', 'gelu']


def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    elif act_str == 'gelu':
        return nn.GELU
    else:
        raise ValueError('invalid activation')

 
class MLP(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='tanh'):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)

        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=True))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=True))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=True))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
                
class LeNet(nn.Sequential):
    def __init__(self, in_channels=1, n_out=10, activation="relu", n_pixels=28):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = nn.Conv2d
        pool = nn.MaxPool2d
        flatten = nn.Flatten(start_dim=1)
        self.add_module("conv1", conv(in_channels, 32, 5, 1))
        self.add_module("pool1", pool(2))
        self.add_module("act1", act())
        self.add_module("conv2", conv(32, 64, 5, 1))
        self.add_module("pool2", pool(2))
        self.add_module("act2", act())
        self.add_module("flatten", flatten)
        self.add_module("lin1", torch.nn.Linear(1024, 128))
        self.add_module("act4", act())
        self.add_module("linout", torch.nn.Linear(128, n_out))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()

# class LeNet(nn.Sequential):
    
#     def __init__(self, in_channels=1, n_out=10, activation='relu', n_pixels=28):
#         super().__init__()
#         mid_kernel_size = 3 if n_pixels == 28 else 5
#         act = get_activation(activation)
#         conv = nn.Conv2d
#         pool = nn.MaxPool2d
#         flatten = nn.Flatten(start_dim=1)
#         self.add_module('conv1', conv(in_channels, 6, 5, 1))
#         self.add_module('act1', act())
#         self.add_module('pool1', pool(2))
#         self.add_module('conv2', conv(6, 16, mid_kernel_size, 1))
#         self.add_module('act2', act())
#         self.add_module('pool2', pool(2))
#         self.add_module('conv3', conv(16, 120, 5, 1))
#         self.add_module('flatten', flatten)
#         self.add_module('act3', act())
#         self.add_module('lin1', torch.nn.Linear(120*1*1, 84))
#         self.add_module('act4', act())
#         self.add_module('linout', torch.nn.Linear(84, n_out))

#     def reset_parameters(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Conv2d, nn.Linear)):
#                 module.reset_parameters()
    
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, augmented=False):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.augmented = augmented
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride, augmented=augmented)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes, augmented=augmented)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            cat_dim = 2 if self.augmented else 1
            identity = torch.cat((identity, torch.zeros_like(identity)), cat_dim)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `3 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, num_classes=10, in_planes=16, in_channels=3):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 6
        layers = 3 * [layer_size]
        self.num_layers = 3 * layer_size
        self.inplanes = in_planes
        self.conv1 = conv3x3(in_channels, in_planes)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.bias2 = Bias()
        self.fc = nn.Linear(in_planes * 4, num_classes)

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, augmented=self.augmented))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, augmented=self.augmented))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(self.bias2(x))
        if self.llc:
            x = self.constant_logit(x)

        return x
