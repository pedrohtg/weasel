import os
import sys
import numpy as np
import torch
import random

from torch.utils import data

from skimage import io
from skimage import util
from skimage import filters
from skimage import measure
from skimage import transform
from skimage import morphology
from skimage import segmentation
from skimage import img_as_float
from skimage import data as skdata

'''
# This class ListDataset use the dataset name and task name to load the respectives images accordingly, modify this class to your use case
# For a simple loader, this version is implemented considering the following folder scheme:

/datasets_root_folder
|--- dataset_1
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
|--- dataset_2
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
...
|--- dataset_n
     |--- images
     |--- ground_truths
     |--- fold_0_img_list.txt

     ...
     |--- fold_k_img_list.txt
'''

# Constants.
root = 'dataset' # The /datasets_root_folder/ in the scheme above

annot_types = ['points', 'regions', 'skels', 'contours', 'grid', 'dense', 'random']

# Class that reads a sequence of image paths from a text file and creates a data.Dataset with them.
class ListDataset(data.Dataset):
    
    def __init__(self, mode, dataset, task, fold, resize_to, num_shots=5, sparsity_mode='dense', sparsity_param=None, imgtype='med'):
        
        assert sparsity_mode in annot_types, "{} annotation type not supported, must be one of following {}.".format(sparsity_mode, annot_types)

        self.imgtype = imgtype
        self.root = root
        self.num_classes = 2

        # Initializing variables.
        self.mode = mode
        self.dataset = dataset
        self.task = task
        self.fold = fold
        self.resize_to = resize_to
        self.num_shots = num_shots
        self.sparsity_mode = sparsity_mode
        self.sparsity_param = sparsity_param
        self.imgtype = imgtype
        
        self.sparsity_mode_list = ['points', 'contours', 'grid', 'regions', 'skels']
        
        # Creating list of paths.
        self.imgs = self.make_dataset()
        
        # Check for consistency in list.
        if len(self.imgs) == 0:
            
            raise (RuntimeError('Found 0 images, please check the data set'))
    
    # Function that create the list of pairs (img_path, mask_path)
    # Adapt this function for your dataset and fold structure
    def make_dataset(self):

        # Making sure the mode is correct.
        assert self.mode in ['train', 'test', 'meta_train', 'meta_test', 'tune_train', 'tune_test']
        items = []
        
        # Setting string for the mode.
        mode_str = ''
        if 'train' in self.mode:
            mode_str = 'trn' if self.imgtype == 'med' else 'train'
        elif 'test' in self.mode:
            mode_str = 'tst' if self.imgtype == 'med' else 'val'

        # Joining input paths.
        img_path = os.path.join(root, self.dataset, 'images')
        msk_path = os.path.join(root, self.dataset, 'ground_truths', self.task)

        # Reading paths from file.
        data_list = []
        data_list = [l.strip('\n') for l in open(os.path.join(self.root, self.dataset, self.task + '_' + mode_str + '_f' + str(self.fold) + '_few_shot.txt')).readlines()]
        
        random.seed(int(self.fold))
        random.shuffle(data_list)

        if self.num_shots != -1 and self.num_shots <= len(data_list):
            data_list = data_list[:self.num_shots]
        
        # Creating list containing image and ground truth paths.
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(msk_path, it))
            items.append(item)
        
        # Returning list.
        return items
    
        
    def sparse_points(self, msk, sparsity='random', index=-1):
        # Linearizing mask.
        msk_ravel = msk.ravel()
        
        # Slicing array for only containing negative class pixels.
        neg_msk = msk_ravel[msk_ravel == 0]
        neg_msk[:] = -1
        
        # Slicing array for only containing positive class pixels.
        pos_msk = msk_ravel[msk_ravel > 0]
        pos_msk[:] = -1
        
        if sparsity == 'random':
            
            # Randomly choosing sparsity (number of points -- [1..20]).
            sparsity = np.random.randint(low=1, high=21)
            
            # Negative mask.
            perm_neg = np.random.permutation(neg_msk.shape[0]) # Random permutation of negative pixels.
            neg_msk[perm_neg[:min(sparsity, len(perm_neg))]] = 0

            # Positive mask.
            perm_pos = np.random.permutation(pos_msk.shape[0]) # Random permutation of positive pixels.
            pos_msk[perm_pos[:min(sparsity, len(perm_pos))]] = 1
        
        elif isinstance(sparsity, int) and sparsity > 0:
            
            # Predetermined sparsity (number of labeled points of each class).
            pass
            
            # Negative mask.
            np.random.seed(index)
            perm_neg = np.random.permutation(neg_msk.shape[0]) # Predetermined permutation of negative pixels (fixed by seed).
            neg_msk[perm_neg[:min(sparsity, len(perm_neg))]] = 0
            
            # Positive mask.
            np.random.seed(index)
            perm_pos = np.random.permutation(pos_msk.shape[0]) # Predetermined permutation of positive pixels (fixed by seed).
            pos_msk[perm_pos[:min(sparsity, len(perm_pos))]] = 1
            
        # Merging negative and positive sparse masks.
        new_msk = np.zeros(msk_ravel.shape[0], dtype=np.int64)
        new_msk[:] = -1
        
        new_msk[msk_ravel == 0] = neg_msk
        new_msk[msk_ravel  > 0] = pos_msk
        
        # Reshaping linearized sparse mask to the original 2 dimensions.
        new_msk = new_msk.reshape(msk.shape)
        
        return new_msk
    
    def sparse_grid(self, msk, sparsity='random', index=-1):
        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:, :] = -1
        
        if sparsity == 'random':
            
            # Random sparsity (x and y point spacing -- [8..20]).
            spacing = (np.random.randint(low=8, high=21),
                       np.random.randint(low=8, high=21))
            
            # Random starting offset.
            starting = (np.random.randint(spacing[0]),
                        np.random.randint(spacing[1]))
            
        elif isinstance(sparsity, int) and sparsity > 0:
            
            # Predetermined sparsity (x and y point spacing).
            spacing = (int(2 ** sparsity),
                       int(2 ** sparsity))
            
            np.random.seed(index)
            
            # Predetermined starting offset (fixed by seed).
            starting = (np.random.randint(spacing[0]),
                        np.random.randint(spacing[1]))
        
        new_msk[starting[0]::spacing[0],
                starting[1]::spacing[1]] = msk[starting[0]::spacing[0],
                                               starting[1]::spacing[1]]
        
        return new_msk
    
    def sparse_contours(self, msk, sparsity='random', index=-1):
        if sparsity == 'random':
            
            # Randomly choosing sparsity (contour proportion -- [0..1]).
            sparsity = np.random.random()
        
            # Random disk radius for erosions and dilations from the original mask.
            radius_dist = np.random.randint(low=3, high=10)

            # Random disk radius for annotation thickness.
            radius_thick = np.random.randint(low=1, high=2)
            
            # Creating morphology elements.
            selem_dist = morphology.disk(radius_dist)
            selem_thick = morphology.disk(radius_thick)

            # Dilating and eroding original mask and obtaining contours.
            msk_neg = morphology.binary_dilation(msk > 0, selem_dist)
            msk_pos = morphology.binary_erosion(msk > 0, selem_dist)

            pos_contr = measure.find_contours(msk_pos, 0.0)
            neg_contr = measure.find_contours(msk_neg, 0.0)

            # Instantiating masks for the boundaries.
            msk_neg_bound = np.zeros_like(msk_neg)
            msk_pos_bound = np.zeros_like(msk_pos)

            # Filling boundary masks.
            for i, obj in enumerate(pos_contr):
                rand_rot = np.random.randint(low=1, high=len(obj)) # Random rotation of contour.
                for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
                    if j < round(len(obj) * sparsity):
                        msk_pos_bound[int(contour[0]), int(contour[1])] = 1

            for i, obj in enumerate(neg_contr):
                rand_rot = np.random.randint(low=1, high=len(obj)) # Random rotation of contour.
                for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
                    if j < round(len(obj) * sparsity):
                        msk_neg_bound[int(contour[0]), int(contour[1])] = 1
            
        elif isinstance(sparsity, float) and sparsity > 0:
            
            # Fixing seed.
            np.random.seed(index)
        
            # Predefined disk radius for erosions and dilations from the original mask (fixed by seed).
            radius_dist = np.random.randint(low=3, high=10)

            # Predefined disk radius for annotation thickness (fixed by seed).
            radius_thick = np.random.randint(low=1, high=2)
            
            # Creating morphology elements.
            selem_dist = morphology.disk(radius_dist)
            selem_thick = morphology.disk(radius_thick)

            # Dilating and eroding original mask and obtaining contours.
            msk_neg = morphology.binary_dilation(msk > 0, selem_dist)
            msk_pos = morphology.binary_erosion(msk > 0, selem_dist)

            pos_contr = measure.find_contours(msk_pos, 0.0)
            neg_contr = measure.find_contours(msk_neg, 0.0)

            # Instantiating masks for the boundaries.
            msk_neg_bound = np.zeros_like(msk_neg)
            msk_pos_bound = np.zeros_like(msk_pos)

            # Filling boundary masks.
            for i, obj in enumerate(pos_contr):
                np.random.seed(index)
                rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
                for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
                    if j < max(1, round(len(obj) * sparsity)):
                        msk_pos_bound[int(contour[0]), int(contour[1])] = 1

            for i, obj in enumerate(neg_contr):
                np.random.seed(index)
                rand_rot = np.random.randint(low=1, high=len(obj)) # Predefined rotation of contour (fixed by seed).
                for j, contour in enumerate(np.roll(obj, rand_rot, axis=0)):
                    if j < max(1, round(len(obj) * sparsity)):
                        msk_neg_bound[int(contour[0]), int(contour[1])] = 1

        # Performing dilation on the boundary masks for adding thickness.
        msk_neg_bound = morphology.dilation(msk_neg_bound, selem=selem_thick)
        msk_pos_bound = morphology.dilation(msk_pos_bound, selem=selem_thick)

        # Grouping positive, negative and negatively labeled pixels.
        new_msk = np.zeros_like(msk, dtype=np.int8)
        new_msk[:] = -1
        new_msk[msk_neg_bound] = 0
        new_msk[msk_pos_bound] = 1

        return new_msk

    def sparse_skels(self, msk, sparsity='random', index=-1):
        if sparsity == 'random':
            sparsity = np.random.random()

        bseed = None # Blobs generator seed 
        if 'tune' in self.mode:
            np.random.seed(index)
            bseed = index

        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Randomly selecting disk radius the annotation thickness.
        radius_thick = np.random.randint(low=1, high=2)
        selem_thick = morphology.disk(radius_thick)

        for c in range(self.num_classes):
                c_msk = (msk == c)
                c_skel = morphology.skeletonize(c_msk)
                c_msk = morphology.binary_dilation(c_skel, selem=selem_thick)

                new_msk[c_msk] = c

        blobs = skdata.binary_blobs(new_msk.shape[0], blob_size_fraction=0.1,
                                  volume_fraction=sparsity, seed=bseed)
        n_sp = np.zeros_like(new_msk)
        n_sp[:] = -1
        n_sp[blobs] = new_msk[blobs]

        return n_sp 
        
    def sparse_region(self, img, msk, sparsity='random', index=-1):
        # Compactness of SLIC for each dataset.
        cpn = {
            # MEDICAL
            'nih_labeled': 0.6,
            'inbreast': 0.6,
            'shenzhen': 0.7,
            'montgomery':0.5,
            'openist':0.5,
            'jsrt':0.5,
            'ufba':0.5,
            'lidc_idri_drr':0.5,
            'panoramic': 0.75,
            'mias': 0.45
        }

        # Copying mask and starting it with -1 for inserting sparsity.
        new_msk = np.zeros_like(msk)
        new_msk[:] = -1

        # Computing SLIC super pixels.
        slic = segmentation.slic(img, n_segments=250, compactness=cpn[self.dataset], start_label=1)
        labels = np.unique(slic)

        # Finding 'pure' regions, that is, the ones that only contain one label within.
        pos_sp = []
        neg_sp = []

        for l in labels:
            sp = msk[slic == l].ravel()
            cnt = np.bincount(sp)
            
            if len(cnt) == 1:
                neg_sp.append(l)
            else:
                if cnt[0] == 0:
                    pos_sp.append(l)

        neg_sp = np.array(neg_sp)
        pos_sp = np.array(pos_sp)
        
        if sparsity == 'random':
            
            # Randomly choosing sparsity (number of slic regions -- [0..1]).
            sparsity = np.random.random()
            
            # Random permutation to negative regions.
            perm_neg = np.random.permutation(len(neg_sp))
            
            # Random permutation to positive regions.
            perm_pos = np.random.permutation(len(pos_sp))
        
        elif isinstance(sparsity, float) and sparsity > 0:
            
            # Fixing seed.
            np.random.seed(index)
            # Predefined permutation to negative regions (fixed by seed).
            perm_neg = np.random.permutation(len(neg_sp))
            
            # Fixing seed.
            np.random.seed(index)
            # Predefined permutation to positive regions (fixed by seed).
            perm_pos = np.random.permutation(len(pos_sp))
        
        # Only keeping the selected k regions.
        for sp in neg_sp[perm_neg[:max(1, round(sparsity * len(perm_neg)))]]: 
            new_msk[slic == sp] = 0
        for sp in pos_sp[perm_pos[:max(1, round(sparsity * len(perm_pos)))]]: 
            new_msk[slic == sp] = 1

        return new_msk

    # Function to load images and masks
    # May need adaptation to your data
    # Returns: img, mask, (path_to_img, img_filename)
    def get_data(self, index):
        img_path, msk_path = self.imgs[index]

        # Reading images.
        img = io.imread(img_path, as_gray=True)
        msk = io.imread(msk_path)
        
        # Removing unwanted channels. For the case of RGB images.
        if len(img.shape) > 2:
            img = img[:, :, 0]
        
        if len(msk.shape) > 2:
            msk = msk[:, :, 0]

        img = img_as_float(img)

        # Resising images.
        if msk.shape[0] > self.resize_to[0] * 2:
            d_res = (self.resize_to[0]*2, self.resize_to[1]*2)
            img = transform.resize(img, d_res, order=1, preserve_range=True)
            msk = transform.resize(msk, d_res, order=0, preserve_range=True)

        img = transform.resize(img, self.resize_to, order=1, preserve_range=True)
        msk = transform.resize(msk, self.resize_to, order=0, preserve_range=True)

        img = img.astype(np.float32)
        msk = msk.astype(np.int64)

        # Binarizing mask.
        if self.num_classes == 2:
            msk[msk != 0] = 1
        
        # Splitting path.
        spl = img_path.split('/')

        return img, msk, spl

    def norm(self, img):
        if len(img.shape) == 2:
            img = (img - img.mean()) / img.std()
        else:
            for b in range(img.shape[2]):
                img[:,:,b] = (img[:,:,b] - img[:,:,b].mean()) / img[:,:,b].std()
        return img

    def torch_channels(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        else:
            img = np.moveaxis(img, -1, 0)
        return img

    def __getitem__(self, index):

        img, msk, spl = self.get_data(index)
            
        if self.sparsity_mode == 'random':
            
            # Randomly selecting sparsity mode.
            sparsity = np.random.randint(0, len(self.sparsity_mode_list))
            
            if sparsity == 0:
                sparse_msk = self.sparse_points(msk, sparsity='random', index=index)
            elif sparsity == 1:
                sparse_msk = self.sparse_contours(msk, sparsity='random', index=index)
            elif sparsity == 2:
                sparse_msk = self.sparse_grid(msk, sparsity='random', index=index)
            elif sparsity == 3:
                sparse_msk = self.sparse_region(img, msk, sparsity='random', index=index)
            elif sparsity == 4:
                sparse_msk = self.sparse_skels( msk, sparsity='random', index=index)
            
        # Randomly selecting sparse points.
        if self.sparsity_mode == 'points':
            sparse_msk = self.sparse_points(msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'contours':
            sparse_msk = self.sparse_contours(msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'grid':
            sparse_msk = self.sparse_grid(msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'regions':
            sparse_msk = self.sparse_region(img, msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'skels':
            sparse_msk = self.sparse_skels(msk, sparsity=self.sparsity_param, index=index)
        elif self.sparsity_mode == 'dense':
            sparse_msk = np.copy(msk)
                
        # Normalization.
        img = self.norm(img)
        
        # Adding channel dimension.
        img = self.torch_channels(img)
        
        # Turning to tensors.
        img = torch.from_numpy(img)
        msk = torch.from_numpy(msk).type(torch.LongTensor)
        

        sparse_msk = torch.from_numpy(sparse_msk).type(torch.LongTensor)
        
        # Returning to iterator.
        return img, msk, sparse_msk, spl[-1]

        
    def __len__(self):

        return len(self.imgs)
