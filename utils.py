import os
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sklearn import metrics
from matplotlib import pyplot as plt

from torchmeta import modules

from collections import OrderedDict

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def prepare_meta_batch(meta_train_set, meta_test_set, index, batch_size=5):
    
    # Acquiring training and test data.
    x_train = []
    y_train = []
    
    x_test = []
    y_test = []
    
    perm_train = torch.randperm(len(meta_train_set[index])).tolist()
    perm_test = torch.randperm(len(meta_test_set[index])).tolist()
    
    for b in range(batch_size):
        
        d_tr = meta_train_set[index][perm_train[b]]
        d_ts = meta_test_set[index][perm_test[b]]
        
        x_tr = d_tr[0].cuda()
        y_tr = d_tr[2].cuda()
        
        x_ts = d_ts[0].cuda()
        y_ts = d_ts[1].cuda()
        
        x_train.append(x_tr)
        y_train.append(y_tr)
        
        x_test.append(x_ts)
        y_test.append(y_ts)
        
    x_train = torch.stack(x_train, dim=0)
    y_train = torch.stack(y_train, dim=0)
    
    x_test = torch.stack(x_test, dim=0)
    y_test = torch.stack(y_test, dim=0)
    
    return x_train, y_train, x_test, y_test

def plot_kernels(kernel, idx, epoch, norm='mean0'):
    if norm == 'mean0':
        tensor = (1/(abs(kernel.min())*2))*kernel + 0.5
    elif norm == '01':
        tensor = (kernel - kernel.min()) / (kernel.max() - kernel.min()) 

    num_kernels = tensor.shape[0]
    num_rows = num_kernels
    num_cols = tensor.shape[1]
    fig = plt.figure(figsize=(16,16))
    fig.tight_layout()
    
    tot = num_rows * num_cols
    pos = range(1, tot+1)

    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            ax1 = fig.add_subplot(num_rows,num_cols,pos[k])
            ax1.imshow(tensor[i][j], cmap='gray')
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            k+=1

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('kernels/kernel' + str(idx) + '_ep' + str(epoch) + '.png', format='png')
    # plt.show()

def accuracy(lab, prd):
    # Obtaining class from prediction.
    prd = prd.argmax(1)
    
    # Tensor to ndarray.
    lab_np = lab.view(-1).detach().cpu().numpy()
    prd_np = prd.view(-1).detach().cpu().numpy()
    
    # Computing metric and returning.
    metric_val = metrics.jaccard_score(lab_np, prd_np)
    
    return metric_val