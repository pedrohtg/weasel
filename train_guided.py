import datetime
import os
import random
import time
import gc
import sys
import numpy as np
import copy
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
from losses import *

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True
seed = 42

settings = Config()
general_params, fewshot_params, task_dicts, args, datainfo, guided_cfg = settings['GENERAL'], settings['FEW-SHOT'], settings['TASKS']['task_dicts'], settings['TRAINING'], settings['DATA'], settings['GUIDED']

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
exp_name = 'guided_nets_multiple_' + conv_name + '_' + data_name + '_' + task_name + '_f' + str(fold_name)

# Type of mapping used by the guided net
mapping = guided_cfg['mapping']

# Defining loss function
loss_name = guided_cfg['loss_fn']

def one_hot_masks(msk):
    
    neg = torch.zeros_like(msk).float()
    pos = torch.zeros_like(msk).float()
    
    neg[msk == 0] = 1
    pos[msk == 1] = 1
    
    neg.unsqueeze_(1)
    pos.unsqueeze_(1)
    
    return neg, pos

def simple_mask(msk):        
    return msk


# Main function.
def main(args):
    
    device = torch.device('cpu')
    if torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda:0')

    # Setting network architecture.
    if (conv_name == 'unet'):
        feat_dim = 32  
        model = UNet(datainfo['num_channels'], datainfo['num_class'], prototype=True).cuda()

        # Creating average pooler.
        pool = nn.AdaptiveAvgPool2d((1, 1))
        pool.to(device)
        
        # Create classifier.
        head = nn.Sequential(nn.Conv2d(feat_dim * 2, feat_dim * 1, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.Conv2d(feat_dim * 1, 2, kernel_size=1))
        nn.init.ones_(head[0].weight)
        nn.init.ones_(head[-1].weight)
        head.to(device)

        if mapping == 'learned':
            label_enc = UNet(1, 2, prototype=True).to(device)
        elif mapping == 'simple':
            label_enc = simple_mask
        else:
            print("Unsuported label encoder function.")
            exit()

    print(model)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

    # Setting optimizer and LR scheduler.
    all_params = list(model.parameters()) + \
                 list(pool.parameters()) + \
                 list(head.parameters())
    if mapping == 'learned':
        all_params += list(label_enc.parameters())
    
    meta_optimizer = optim.Adam(all_params, args['lr'])
    
    # Setting scheduler.
    scheduler = optim.lr_scheduler.StepLR(meta_optimizer, args['epoch_num'] // 4, gamma=0.5, last_epoch=-1)

    # Loading optimizer state in case of resuming training.
    if args['snapshot'] == '':

        curr_epoch = 1

    else:

        print('Training resuming from epoch ' + str(args['snapshot']) + '...')
        model.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'meta.pth')))
        pool.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'pool.pth')))
        head.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'head.pth')))
        if isinstance(label_enc, nn.Module):
            label_enc.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'model_lbl.pth')))
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
        meta_train_test(meta_train_set, meta_test_set, model, label_enc, pool, head,
                        loss_name, meta_optimizer, 
                        epoch, device, epoch % args['test_freq'] == 0, args)
        
        if epoch % args['test_freq'] == 0:
            # Saving model and optimizer.
            model_ckpt_path = os.path.join(ckpt_path, exp_name, 'meta.pth')
            model_lbl_ckpt_path = os.path.join(ckpt_path, exp_name, 'model_lbl.pth')
            pool_ckpt_path = os.path.join(ckpt_path, exp_name, 'pool.pth')
            head_ckpt_path = os.path.join(ckpt_path, exp_name, 'head.pth')
            optim_ckpt_path = os.path.join(ckpt_path, exp_name, 'opt_meta.pth')
            
            torch.save(model, model_ckpt_path)
            if isinstance(label_enc, nn.Module):
                torch.save(label_enc, model_lbl_ckpt_path)
            torch.save(pool, pool_ckpt_path)
            torch.save(head, head_ckpt_path)
            torch.save(meta_optimizer, optim_ckpt_path)

            run_sparse_tuning(loader_dict, model, pool, head, label_enc, loss_name, meta_optimizer, epoch, device, task_name)
            
        scheduler.step()

def fast_adapt(sup_img,
               sup_msk,
               qry_img,
               qry_msk,
               model,
               model_lbl,
               pool,
               head,
               loss_name,
               device):
    
    # Adapt the model.
    weights = loss_weights(sup_msk, device)
    
    # Negative and positive encodings of support.
    sup_neg, sup_pos = one_hot_masks(sup_msk)
    
    # Computing query prior embeddings.
    qry_img_emb = model(qry_img)
    
    # Computing support embeddings for image and negative/positive encodings.
    sup_img_emb = model(sup_img)
    sup_neg_emb = model_lbl(sup_neg)
    sup_pos_emb = model_lbl(sup_pos)
    
    if mapping == 'simple':
        sup_cat_emb = sup_img_emb * (sup_neg_emb + sup_pos_emb) #(sup_img_emb * sup_neg_emb) * sup_pos_emb
    else:
        sup_cat_emb = (sup_img_emb * sup_neg_emb) * sup_pos_emb
    sup_cat_emb = pool(sup_cat_emb)
    sup_cat_emb = torch.mean(sup_cat_emb, dim=0, keepdim=True)
    
    # Tiling z on batch and spatial dimensions.
    sup_emb = torch.tile(sup_cat_emb, (qry_img_emb.shape[0],
                                       1,
                                       qry_img_emb.shape[2],
                                       qry_img_emb.shape[3]))
    
    # Computing query posterior embeddings.
    qry_emb = torch.cat([sup_emb, qry_img_emb], 1)
    
    # Predicting for query.
    qry_prd = head(qry_emb)
    
    # Computing supervised loss on query predictions.
    qry_err = loss_fn(qry_prd, qry_msk, weights, loss_name, device)
    qry_met = accuracy(qry_msk, qry_prd)
    
    return qry_err, qry_met

# Training function.
def meta_train_test(meta_train_set, meta_test_set, model, model_lbl, pool, head, loss_name, meta_optimizer, epoch, device, save_model, args):
    
    # Setting network for training mode.
    model.train()
    head.train()
    if isinstance(model_lbl, nn.Module):
        model_lbl.train()
    
    # List for batch losses.
    train_outer_loss_list = list()
    met_list = list()
    
    num_tasks = len(meta_train_set)
            
    # Initiating counters.
    meta_qry_err = 0.0
    meta_qry_met = 0.0
    
    # Inner loop iterations.
    inner_count = 0
    
    # Randomly selecting tasks.
    perm = np.random.permutation(num_tasks)
    print('Ep: ' + str(epoch) + ', task subset: ' + str(perm[:args['n_metatasks_iter']]))
    sys.stdout.flush()
        
    indices = perm[:args['n_metatasks_iter']]
    
    for index in indices:
        # Acquiring training and test data.
        x_train = []
        y_train = []
        
        x_test = []
        y_test = []
        
        x_tr, y_tr, x_ts, y_ts = prepare_meta_batch(meta_train_set, meta_test_set, index, args['batch_size'])
        
        x_train.append(x_tr)
        y_train.append(y_tr)
        
        x_test.append(x_ts)
        y_test.append(y_ts)
    
        # Concatenating tensors.
        x_train = torch.cat(x_train, dim=0)
        y_train = torch.cat(y_train, dim=0)
        
        x_test = torch.cat(x_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        
        # Clearing model gradients.
        model.zero_grad()
        if isinstance(model_lbl, nn.Module):
            model_lbl.zero_grad()
        
        ##########################################################################
        # Start of prototyping. ##################################################
        ##########################################################################
        
        # Compute meta-training loss.
        qry_err, qry_met = fast_adapt(x_train,
                                      y_train,
                                      x_test,
                                      y_test,
                                      model,
                                      model_lbl,
                                      pool,
                                      head,
                                      loss_name,
                                      device)
        
        if not torch.any(torch.isnan(qry_err)):
            
            # Backpropagating.
            qry_err.backward()
            
            # Taking optimization step.
            meta_optimizer.step()
            
            # Updating counters.
            meta_qry_err += qry_err.item()
            meta_qry_met += qry_met
            
            inner_count += 1

    # Updating lists.
    meta_qry_err /= inner_count
    meta_qry_met /= inner_count
    
    train_outer_loss_list.append(meta_qry_err)
    met_list.append(meta_qry_met)
    
    # Saving meta-model.
    if save_model:
        torch.save(model.state_dict(), os.path.join(ckpt_path, exp_name, 'meta.pth'))
        torch.save(meta_optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_meta.pth'))
        
    # Printing epoch loss.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [train loss %.4f]' % (
        epoch, np.asarray(train_outer_loss_list).mean()))
    # Print loss and accuracy.
    print('  Meta Val Err', meta_qry_err)
    print('  Meta Val Jac', meta_qry_met)
    print('--------------------------------------------------------------------')
    sys.stdout.flush()
    
def run_sparse_tuning(loader_dict, model, pool, head, model_lbl, loss_name, meta_optimizer, epoch, device, task_name):
    
    # Tuning/testing on points.
    for dict_points in loader_dict['points']:

        n_shots = dict_points['n_shots']
        sparsity = dict_points['sparsity']

        print('    Evaluating \'points\' (%d-shot, %d-points)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_points['train'], dict_points['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args, 
                        task_name, 'points_(%d-shot_%d-points)' % (n_shots, sparsity))

    # Tuning/testing on contours.
    for dict_contours in loader_dict['contours']:

        n_shots = dict_contours['n_shots']
        sparsity = dict_contours['sparsity']

        print('    Evaluating \'contours\' (%d-shot, %.2f-density)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_contours['train'], dict_contours['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args,
                        task_name, 'contours_(%d-shot_%.2f-density)' % (n_shots, sparsity))

    # Tuning/testing on grid.
    for dict_grid in loader_dict['grid']:

        n_shots = dict_grid['n_shots']
        sparsity = dict_grid['sparsity']

        print('    Evaluating \'grid\' (%d-shot, %d-spacing)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_grid['train'], dict_grid['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args,
                        task_name, 'grid_(%d-shot_%d-spacing)' % (n_shots, sparsity))

    # Tuning/testing on regions.
    for dict_regions in loader_dict['regions']:

        n_shots = dict_regions['n_shots']
        sparsity = dict_regions['sparsity']

        print('    Evaluating \'regions\' (%d-shot, %.2f-regions)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_regions['train'], dict_regions['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args,
                        task_name, 'regions_(%d-shot_%.2f-regions)' % (n_shots, sparsity))

    # Tuning/testing on skels.
    for dict_skels in loader_dict['skels']:

        n_shots = dict_skels['n_shots']
        sparsity = dict_skels['sparsity']

        print('    Evaluating \'skels\' (%d-shot, %.2f-skels)...' % (n_shots, sparsity))
        sys.stdout.flush()

        tune_train_test(dict_skels['train'], dict_skels['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args,
                        task_name, 'skels_(%d-shot_%.2f-skels)' % (n_shots, sparsity))

    # Tuning/testing on dense.
    for dict_dense in loader_dict['dense']:

        n_shots = dict_dense['n_shots']

        print('    Evaluating \'dense\' (%d-shot)...' % (n_shots))
        sys.stdout.flush()

        tune_train_test(dict_dense['train'], dict_dense['test'], model, model_lbl, pool, head, loss_name,
                        meta_optimizer, epoch, device, args,
                        task_name, 'dense_(%d-shot)' % (n_shots))

def tune_train_test(tune_train_loader, tune_test_loader, model, model_lbl, pool, head, loss_name, meta_optimizer, epoch, device, args, task_name, sparsity_mode):
    
    # Creating output directories.
    if epoch == args['epoch_num']:
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch)))
        check_mkdir(os.path.join(outp_path, exp_name, sparsity_mode + '_test_epoch_' + str(epoch)))
    
    with torch.no_grad():
    
        # Setting network for training mode.
        model.eval()
        head.eval()

        # Zeroing model gradient.
        model.zero_grad()
        head.zero_grad()

        # Creating lists for tune train embeddings and labels.
        x_train_list = []
        y_train_list = []

        # Iterating over tuning train batches.
        for i, data in enumerate(tune_train_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            x_tr, y_dense, y_tr, img_name = data

            # Casting tensors to cuda.
            x_tr = x_tr.cuda()
            y_tr = y_tr.cuda()

            x_train_list.append(x_tr)
            y_train_list.append(y_tr)

        x_tr = torch.cat(x_train_list, 0)
        y_tr = torch.cat(y_train_list, 0)
        
        label_enc = copy.deepcopy(model_lbl) if isinstance(model_lbl, nn.Module) else model_lbl

        # Tuning and saving outputs.
        err_list, met_list = fast_tune(x_tr.to(device),
                                       y_tr.to(device),
                                       tune_test_loader,
                                       sparsity_mode,
                                       epoch,
                                       copy.deepcopy(model),
                                       label_enc,
                                       copy.deepcopy(pool),
                                       copy.deepcopy(head),
                                       loss_name,
                                       device,
                                       os.path.join(outp_path, exp_name))

    if epoch == args['epoch_num']:

        # Iterating over tuning train batches for saving.
        for i, data in enumerate(tune_train_loader):

            # Obtaining images, dense labels, sparse labels and paths for batch.
            _, y_dense, y_sparse, img_name = data

            for j in range(len(img_name)):

                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch), img_name[j].replace('.png', '_dense.png')), skimage.img_as_ubyte(y_dense[j].cpu().squeeze().numpy() * 255))
                io.imsave(os.path.join(outp_path, exp_name, sparsity_mode + '_train_epoch_' + str(epoch), img_name[j].replace('.png', '_sparse.png')), skimage.img_as_ubyte((y_sparse[j].cpu().squeeze().numpy() + 1) * 127))
                
def fast_tune(sup_img,
              sup_msk,
              qry_loader,
              sparsity_mode,
              epoch,
              model,
              model_lbl,
              pool,
              head,
              loss_name,
              device,
              exp_dir_qry):
    
    # Setting model to evaluation mode (no tuning required on Guided Nets).
    model.eval()
    head.eval()
    with torch.no_grad():
        sup_emb = None

        # Adapt the model.
        weights = loss_weights(sup_msk, device)
        
        # Negative and positive encodings of support.
        sup_neg, sup_pos = one_hot_masks(sup_msk)
        
        # Computing support embeddings for image and negative/positive encodings.
        sup_img_emb = model(sup_img)
        sup_neg_emb = model_lbl(sup_neg)
        sup_pos_emb = model_lbl(sup_pos)
        
        if mapping == 'simple':
            sup_cat_emb = sup_img_emb * (sup_neg_emb + sup_pos_emb)
        else:
            sup_cat_emb = (sup_img_emb * sup_neg_emb) * sup_pos_emb
        
        sup_cat_emb = pool(sup_cat_emb)
        sup_cat_emb = torch.mean(sup_cat_emb, dim=0, keepdim=True)
        
        # Tiling z on batch and spatial dimensions.
        sup_emb = torch.tile(sup_cat_emb, (1,
                                           1,
                                           sup_img.shape[2],
                                           sup_img.shape[3]))
        

        # Evaluate the adapted model.
        qry_err_list = []
        qry_met_list = []
        
        # To evaluate the model
        labs_all, prds_all = [], []
        labs_np, prds_np = None, None

        # Iterating over query batches.
        for batch in qry_loader:
            qry_emb, qry_prd = None, None
            qry_img, qry_msk, _, qry_names = batch

            qry_img = qry_img.to(device)
            qry_msk = qry_msk.to(device)

            # Computing query prior embeddings.
            qry_img_emb = model(qry_img)
            
            # Computing query posterior embeddings.
            qry_emb = torch.cat([sup_emb, qry_img_emb], 1)
            
            # Predicting for query.
            qry_prd = head(qry_emb)

            # qry_prd = torch.rand_like(qry_prd)

            labs_all.append(qry_msk.cpu().numpy().squeeze())
            prds_all.append(qry_prd.argmax(1).cpu().numpy().squeeze())
            
            # Computing supervised loss on query predictions.
            qry_err = loss_fn(qry_prd, qry_msk, weights, loss_name, device)
            qry_met = accuracy(qry_msk, qry_prd)

            if torch.any(torch.isnan(qry_err)):

                qry_err_list.append(0.0)
                qry_met_list.append(0.0)

                qry_prd = torch.zeros((2, qry_msk.shape[-2], qry_msk.shape[-1]), dtype=torch.float32)
                qry_prd[0, :, :] = 1.0

            else:

                qry_err_list.append(qry_err.item())
                qry_met_list.append(qry_met)
            
            # Saving query image, mask and prediction.
            if epoch == args['epoch_num']:
                io.imsave(os.path.join(exp_dir_qry, sparsity_mode + '_test_epoch_' + str(epoch), qry_names[0]), 
                                       skimage.img_as_ubyte(qry_prd.argmax(1).cpu().numpy().astype(np.uint8).squeeze() * 255))


        qry_err_np = np.asarray(qry_err_list)
        qry_met_np = np.asarray(qry_met_list)

        # Converting to numpy for computing metrics.
        labs_np = np.asarray(labs_all).ravel()
        prds_np = np.asarray(prds_all).ravel()

        # Computing metrics.
        iou = metrics.jaccard_score(labs_np, prds_np)

        # Printing metric.
        print('--------------------------------------------------------------------')
        print('Jaccard test "%s": %.2f' % (sparsity_mode, iou * 100))
        print('--------------------------------------------------------------------')

        print('    Error: %.2f +/- %.2f' % (qry_err_np.mean(), qry_err_np.std()))
        print('    Metric: %.2f +/- %.2f' % (qry_met_np.mean(), qry_met_np.std()))
       
    # Reverting model to training mode.
    model.train()
    head.train()
    
    return qry_err_list, qry_met_list


if __name__ == '__main__':
    main(args)
