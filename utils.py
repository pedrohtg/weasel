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

def get_tune_loaders(shots, points, contours, grid, regions, skels, data_name, task_name, fold_name, resize_to, args):
    
    # Tuning and testing on sparsity mode 'points'.
    points_loader = []
    for n_shots in shots:
        for sparsity in points:
            
            tune_train_points_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='points', sparsity_param=sparsity, imgtype=args['imgtype'])
            tune_train_points_loader = DataLoader(tune_train_points_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_points_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
            tune_test_points_loader = DataLoader(tune_test_points_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
            
            points_loader.append({
                'n_shots': n_shots,
                'sparsity': sparsity,
                'train': tune_train_points_loader,
                'test': tune_test_points_loader
            })
            
    # Tuning and testing on sparsity mode 'contours'.
    contours_loader = []
    for n_shots in shots:
        for sparsity in contours:
            
            tune_train_contours_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='contours', sparsity_param=sparsity, imgtype=args['imgtype'])
            tune_train_contours_loader = DataLoader(tune_train_contours_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_contours_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
            tune_test_contours_loader = DataLoader(tune_test_contours_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
            
            contours_loader.append({
                'n_shots': n_shots,
                'sparsity': sparsity,
                'train': tune_train_contours_loader,
                'test': tune_test_contours_loader
            })
            
    # Tuning and testing on sparsity mode 'grid'.
    grid_loader = []
    for n_shots in shots:
        for sparsity in grid:
            
            tune_train_grid_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='grid', sparsity_param=sparsity, imgtype=args['imgtype'])
            tune_train_grid_loader = DataLoader(tune_train_grid_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_grid_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
            tune_test_grid_loader = DataLoader(tune_test_grid_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
            
            grid_loader.append({
                'n_shots': n_shots,
                'sparsity': sparsity,
                'train': tune_train_grid_loader,
                'test': tune_test_grid_loader
            })
            
    # Tuning and testing on sparsity mode 'regions'.
    regions_loader = []
    for n_shots in shots:
        for sparsity in regions:
            
            tune_train_regions_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='regions', sparsity_param=sparsity, imgtype=args['imgtype'])
            tune_train_regions_loader = DataLoader(tune_train_regions_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)

            tune_test_regions_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
            tune_test_regions_loader = DataLoader(tune_test_regions_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
            
            regions_loader.append({
                'n_shots': n_shots,
                'sparsity': sparsity,
                'train': tune_train_regions_loader,
                'test': tune_test_regions_loader
            })

    # Tuning and testing on sparsity mode 'skels'.
    skels_loader = []
    for n_shots in shots:
        for sparsity in skels:
            
            tune_train_skels_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='skels', sparsity_param=sparsity, imgtype=args['imgtype'])
            tune_train_skels_loader = DataLoader(tune_train_skels_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_skels_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
            tune_test_skels_loader = DataLoader(tune_test_skels_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
            
            skels_loader.append({
                'n_shots': n_shots,
                'sparsity': sparsity,
                'train': tune_train_skels_loader,
                'test': tune_test_skels_loader
            })
            
    # Tuning and testing on sparsity mode 'dense'.
    dense_loader = []
    for n_shots in shots:
        
        # Setting dense dataset.
        tune_train_dense_set = list_dataset.ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='dense', imgtype=args['imgtype'])
        tune_train_dense_loader = DataLoader(tune_train_dense_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
        
        tune_test_dense_set = list_dataset.ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=args['imgtype'])
        tune_test_dense_loader = DataLoader(tune_test_dense_set, batch_size=1, num_workers=args['num_workers'], shuffle=False)
        
        dense_loader.append({
            'n_shots': n_shots,
            'train': tune_train_dense_loader,
            'test': tune_test_dense_loader
        })
        
    return {'points': points_loader,
            'contours': contours_loader,
            'grid': grid_loader,
            'regions': regions_loader,
            'skels': skels_loader,
            'dense': dense_loader}