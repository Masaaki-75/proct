import torch
from torch import nn


def init_weights_with_scale(modules, scale=1):
    if not isinstance(modules, list):
        modules = [modules]
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias.data, 0.0)
