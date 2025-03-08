

import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import random
import os
import time
import json
import glob

from dataset import get_data_transforms



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))

    return loss


def gen_mask(C, HW):
    mask = torch.zeros(C,HW,HW)
    _m_ = torch.zeros(HW,HW)
    counter = 0
    for i in range(HW):
        counter += 1
        for j in range(HW):
            counter += 1
            if counter % 2 == 0:
                _m_[i][j] = 1

    # Alternating 0-1 mask across channels
    for i in range(C):
        if i % 2 == 0:
            mask[i] = _m_
        else:
            mask[i] = 1 - _m_   
            
    return mask