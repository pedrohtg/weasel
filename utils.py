import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchmeta import modules

from collections import OrderedDict

import list_dataset
from torch.utils.data import DataLoader

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, modules.MetaConv2d) or isinstance(module, modules.MetaLinear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(module, modules.MetaBatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()