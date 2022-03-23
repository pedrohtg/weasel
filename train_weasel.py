import datetime
import os
import random
import time
import gc
import sys
import numpy as np
import skimage
import csv

from skimage import io

from sklearn import metrics

import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import Config
from data import list_dataset, list_loader
from models import *
from utils import *

cudnn.benchmark = True

settings = Config()
general_params, fewshot_params, task_dicts, args, datainfo = settings['GENERAL'], settings['FEW-SHOT'], settings['TASKS']['task_dicts'], settings['TRAINING'], settings['DATA']

# Predefining directories.
ckpt_path = general_params['ckpt_path']
outp_path = general_params['outp_path']

# Reading system parameters.
conv_name = general_params['model']
data_name = general_params['dataset']
task_name = general_params['task']
fold_name = general_params['fold']

# Sparsity parameters.
listLoader = list_loader.ListLoader(fewshot_params)

# Setting experiment name.
exp_name = 'maml_multiple_' + conv_name + '_' + data_name + '_' + task_name + '_f' + str(fold_name)

# Main function.
def main(args):

    # Setting network architecture.
    if (conv_name == 'unet'):

        net = UNet(datainfo['num_channels'], datainfo['num_class']).cuda()

    print(net)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('# of parameters: ' + str(n_params))
    sys.stdout.flush()
    
    resize_to = (args['h_size'], args['w_size'])
    
    # Setting meta datasets.
    print('Setting meta-dataset loaders...')
    sys.stdout.flush()
    meta_train_set = [list_dataset.ListDataset('meta_train', d['domain'], d['task'], fold_name, resize_to, num_shots=-1, sparsity_mode='random', imgtype=datainfo['imgtype']) for d in task_dicts if d['domain'] != data_name or d['task'] != task_name]
    
    meta_test_set = [list_dataset.ListDataset('meta_test', d['domain'], d['task'], fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=datainfo['imgtype']) for d in task_dicts if d['domain'] != data_name or d['task'] != task_name]
    
    # Setting tuning and testing loaders.
    print('Setting tuning loaders...')
    sys.stdout.flush()
    loader_dict = listLoader.get_loaders(
        data_name,
        task_name,
        fold_name,
        resize_to,
        args, imgtype=datainfo['imgtype'])
    
    # Setting optimizer.
    meta_optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], betas=(args['momentum'], 0.99))
    
    # Setting scheduler.
    scheduler = optim.lr_scheduler.StepLR(meta_optimizer, args['lr_scheduler_step_size'], gamma=args['lr_scheduler_gamma'], last_epoch=-1)
    
    # Loading optimizer state in case of resuming training.
    if args['snapshot'] == '':
        curr_epoch = 1

    else:
        print('Training resuming from epoch ' + str(args['snapshot']) + '...')
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'meta.pth')))
        meta_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_meta.pth')))
        curr_epoch = int(args['snapshot']) + 1
    
    # Making sure checkpoint and output directories are created.
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    check_mkdir(outp_path)
    check_mkdir(os.path.join(outp_path, exp_name))
    
    # Iterating over epochs.
    for epoch in range(curr_epoch, args['epoch_num'] + 1):
        
        # Meta training on source datasets.
        meta_train_test(meta_train_set, meta_test_set, net, meta_optimizer, epoch, epoch % args['test_freq'] == 0, args)
        
        if epoch % args['test_freq'] == 0:
            
            run_sparse_tuning(loader_dict, net, meta_optimizer, epoch, task_name)
            
        scheduler.step()

# Training function.
def meta_train_test(meta_train_set, meta_test_set, net, meta_optimizer, epoch, save_model, args):
    
    # Setting network for training mode.
    net.train()
    
    # List for batch losses.
    train_outer_loss_list = list()
    
    num_tasks = len(meta_train_set)
    
    n_batches = 5
    
    # Iterating over batches.
    for i in range(n_batches):
        
        # Randomly selecting tasks.
        perm = np.random.permutation(num_tasks)
        print('Ep: ' + str(epoch) + ', it: ' + str(i + 1) + ', task subset: ' + str(perm[:args['n_metatasks_iter']]))
        sys.stdout.flush()
        
        indices = perm[:args['n_metatasks_iter']]
        
        # Acquiring training and test data.
        x_train = []
        y_train = []
        
        x_test = []
        y_test = []
        
        for index in indices:
            
            x_tr, y_tr, x_ts, y_ts = prepare_meta_batch(meta_train_set, meta_test_set, index, args['batch_size'])
            
            x_train.append(x_tr)
            y_train.append(y_tr)
            
            x_test.append(x_ts)
            y_test.append(y_ts)
        
        ##########################################################################
        # Outer loop. ############################################################
        ##########################################################################
        
        # Clearing model gradients.
        net.zero_grad()
        
        # Resetting outer loss.
        outer_loss = torch.tensor(0.0).cuda()
        
        # Iterating over tasks.
        for j in range(len(x_train)):
            
            x_tr = x_train[j]
            y_tr = y_train[j]
            
            x_ts = x_test[j]
            y_ts = y_test[j]
        
            ######################################################################
            # Inner loop. ########################################################
            ######################################################################
            
            # Forwarding through model.
            p_tr = net(x_tr)
            
            # Computing inner loss.
            inner_loss = F.cross_entropy(p_tr, y_tr, ignore_index=-1)
            
            # Zeroing model gradient.
            net.zero_grad()
            
            # Computing metaparameters.
            params = update_parameters(net, inner_loss, step_size=args['step_size'], first_order=args['first_order'])
            
            # Verifying performance on task test set.
            p_ts = net(x_ts, params=params)
            
            # Accumulating outer loss.
            outer_loss += F.cross_entropy(p_ts, y_ts, ignore_index=-1)
            
            ######################################################################
            # End of inner loop. #################################################
            ######################################################################
        
        # Clears the gradients of meta_optimizer.
        meta_optimizer.zero_grad()
        
        # Computing loss.
        outer_loss.div_(len(x_test))
        
        # Computing backpropagation.
        outer_loss.backward()
        meta_optimizer.step()
        
        # Updating loss meter.
        train_outer_loss_list.append(outer_loss.detach().item())
            
        ##########################################################################
        # End of outer loop. #####################################################
        ##########################################################################
    
    # Saving meta-model.
    if save_model:
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, 'meta.pth'))
        torch.save(meta_optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_meta.pth'))
    
    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [train loss %.4f]' % (
        epoch, np.asarray(train_outer_loss_list).mean()))
    print('--------------------------------------------------------------------')
    sys.stdout.flush()
    
def run_sparse_tuning(loader_dict, net, meta_optimizer, epoch, task_name):
    
    # Tuning/testing on points.
    for dict_points in loader_dict['points']:

        n_shots = dict_points['n_shots']
        sparsity = dict_points['sparsity']

        print('    Evaluating \'points\' (%d-shot, %d-points)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_points['train'], dict_points['test'], net, meta_optimizer, epoch, args, task_name, 'points_(%d-shot_%d-points)' % (n_shots, sparsity))

    # Tuning/testing on contours.
    for dict_contours in loader_dict['contours']:

        n_shots = dict_contours['n_shots']
        sparsity = dict_contours['sparsity']

        print('    Evaluating \'contours\' (%d-shot, %.2f-density)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_contours['train'], dict_contours['test'], net, meta_optimizer, epoch, args, task_name, 'contours_(%d-shot_%.2f-density)' % (n_shots, sparsity))

    # Tuning/testing on grid.
    for dict_grid in loader_dict['grid']:

        n_shots = dict_grid['n_shots']
        sparsity = dict_grid['sparsity']

        print('    Evaluating \'grid\' (%d-shot, %d-spacing)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_grid['train'], dict_grid['test'], net, meta_optimizer, epoch, args, task_name, 'grid_(%d-shot_%d-spacing)' % (n_shots, sparsity))

    # Tuning/testing on regions.
    for dict_regions in loader_dict['regions']:

        n_shots = dict_regions['n_shots']
        sparsity = dict_regions['sparsity']

        print('    Evaluating \'regions\' (%d-shot, %.2f-regions)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_regions['train'], dict_regions['test'], net, meta_optimizer, epoch, args, task_name, 'regions_(%d-shot_%.2f-regions)' % (n_shots, sparsity))


    # Tuning/testing on skels.
    for dict_skels in loader_dict['skels']:

        n_shots = dict_skels['n_shots']
        sparsity = dict_skels['sparsity']

        print('    Evaluating \'skels\' (%d-shot, %.2f-skels)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_skels['train'], dict_skels['test'], net, meta_optimizer, epoch, args, task_name, 'skels_(%d-shot_%.2f-skels)' % (n_shots, sparsity))


    # Tuning/testing on dense.
    for dict_dense in loader_dict['dense']:

        n_shots = dict_dense['n_shots']

        print('    Evaluating \'dense\' (%d-shot)...' % (n_shots))
        sys.stdout.flush()

        tune_train_test(dict_dense['train'], dict_dense['test'], net, meta_optimizer, epoch, args, task_name, 'dense_(%d-shot)' % (n_shots))

def tune_train_test(tune_train_loader, tune_test_loader, net, meta_optimizer, epoch, args, task_name, sparsity_mode):
    
    # Creating output directories.
    if epoch == args['epoch_num']:
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch)))
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_test_epoch_' + str(epoch)))
    
    # Setting network for training mode.
    net.train()
    
    # Zeroing model gradient.
    net.zero_grad()
    
    # Repeatedly cycling over batches.
    for c in range(args['tuning_epochs']):
        
        print('Tuning epoch %d/%d' % (c + 1, args['tuning_epochs']))
        sys.stdout.flush()
        
        # Iterating over tuning train batches.
        for i, data in enumerate(tune_train_loader):
            
            # Obtaining images, dense labels, sparse labels and paths for batch.
            x_tr, y_dense, y_tr, img_name = data
            
            # Casting tensors to cuda.
            x_tr, y_tr = x_tr.cuda(), y_tr.cuda()
            
            # Casting to cuda variables.
            x_tr = Variable(x_tr).cuda()
            y_tr = Variable(y_tr).cuda()
            
            # Zeroing gradients for optimizer.
            meta_optimizer.zero_grad()
            
            # Forwarding through model.
            p_tr = net(x_tr)
            
            # Computing inner loss.
            tune_train_loss = F.cross_entropy(p_tr, y_tr, ignore_index=-1)
            
            # Computing gradients and taking step in optimizer.
            tune_train_loss.backward()
            meta_optimizer.step()
            
        if (c + 1) % args['tuning_freq'] == 0:
                
            ##########################################
            # Starting test. #########################
            ##########################################
            
            with torch.no_grad():
                
                # Setting network for evaluation mode.
                net.eval()
                
                # Initiating lists for labels and predictions.
                labs_all, prds_all = [], []
                
                # Iterating over tuning test batches.
                for i, data in enumerate(tune_test_loader):
                    
                    # Obtaining images, labels and paths for batch.
                    x_ts, y_ts, _, img_name = data
                    
                    # Casting to cuda variables.
                    x_ts = Variable(x_ts, volatile=True).cuda()
                    y_ts = Variable(y_ts, volatile=True).cuda()
                    
                    # Forwarding.
                    p_ts = net(x_ts)
                    
                    # Obtaining predictions.
                    prds = p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
                    
                    # Appending data to lists.
                    labs_all.append(y_ts.detach().squeeze(0).cpu().numpy())
                    prds_all.append(prds)
                    
                # Converting to numpy for computing metrics.
                labs_np = np.asarray(labs_all).ravel()
                prds_np = np.asarray(prds_all).ravel()
                
                # Computing metrics.
                iou = metrics.jaccard_score(labs_np, prds_np)
                
                print('Jaccard test "%s" %d/%d: %.2f' % (sparsity_mode, c + 1, args['tuning_epochs'], iou * 100,))
                sys.stdout.flush()
                
            ##########################################
            # Finishing test. ########################
            ##########################################
                
    if epoch == args['epoch_num']:
        
        # Iterating over tuning train batches for saving.
        for i, data in enumerate(tune_train_loader):
            
            # Obtaining images, dense labels, sparse labels and paths for batch.
            _, y_dense, y_sparse, img_name = data
            
            for j in range(len(img_name)):
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch), img_name[j].replace('.png', '_dense.png')), skimage.img_as_ubyte(y_dense[j].cpu().squeeze().numpy() * 255))
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch), img_name[j].replace('.png', '_sparse.png')), skimage.img_as_ubyte((y_sparse[j].cpu().squeeze().numpy() + 1) * 127))
                
    # List for batch losses.
    tune_test_loss_list = list()
    
    # Initiating lists for images, labels and predictions.
    inps_all, labs_all, prds_all = [], [], []
    
    with torch.no_grad():
        
        # Setting network for evaluation mode.
        net.eval()
        
        # Iterating over tuning test batches.
        for i, data in enumerate(tune_test_loader):
            
            # Obtaining images, labels and paths for batch.
            x_ts, y_ts, _, img_name = data
            
            # Casting to cuda variables.
            x_ts = Variable(x_ts, volatile=True).cuda()
            y_ts = Variable(y_ts, volatile=True).cuda()
            
            # Forwarding.
            p_ts = net(x_ts)
            
            # Computing loss.
            tune_test_loss = F.cross_entropy(p_ts, y_ts, ignore_index=-1)
            
            # Obtaining predictions.
            prds = p_ts.detach().max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            
            # Appending data to lists.
            inps_all.append(x_ts.detach().squeeze(0).cpu())
            labs_all.append(y_ts.detach().squeeze(0).cpu().numpy())
            prds_all.append(prds)
            
            # Updating loss meter.
            tune_test_loss_list.append(tune_test_loss.detach().item())
            
            # Saving predictions.
            if epoch == args['epoch_num']:
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_test_epoch_' + str(epoch), img_name[0]), skimage.img_as_ubyte(prds * 255))
    
    # Converting to numpy for computing metrics.
    labs_np = np.asarray(labs_all).ravel()
    prds_np = np.asarray(prds_all).ravel()
    
    # Computing metrics.
    iou = metrics.jaccard_score(labs_np, prds_np)
    
    # Printing metric.
    print('--------------------------------------------------------------------')
    print('Jaccard test "%s" %d/%d: %.2f' % (sparsity_mode, args['tuning_epochs'], args['tuning_epochs'], iou * 100))
    print('--------------------------------------------------------------------')
    sys.stdout.flush()
    
    # Loading model.
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'meta.pth')))
    meta_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_meta.pth')))

if __name__ == '__main__':
    main(args)