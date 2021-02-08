import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torchmeta import modules

from collections import OrderedDict

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
