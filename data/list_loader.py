from .list_dataset import ListDataset
from torch.utils.data import DataLoader

class ListLoader(object):
    def __init__(self, params=None):
        if params is None: # Default
            self.list_shots = [1, 5, 10, 20]                                 # Number of shots in the task (i.e, total annotated sparse samples)
            self.list_sparsity_points = [1, 5, 10, 20]                       # Number of labeled pixels in point annotation
            self.list_sparsity_contours = [0.05, 0.10, 0.25, 0.50, 1.00]     # Density of the contours (1, is the complete contours)
            self.list_sparsity_grid = [8, 12, 16, 20]                        # Spacing between selected pixels in grid annotation
            self.list_sparsity_regions = [0.05, 0.10, 0.25, 0.50, 1.00]      # Percentege of regions labeled (1, all \pure\ regions are labeled)
            self.list_sparsity_skels = [0.05, 0.10, 0.25, 0.50, 1.00]        # Density of the skeletons (1, is the complete skeletons)
        else:
            self.list_shots = params['list_shots']
            self.list_sparsity_points = params['list_sparsity_points']
            self.list_sparsity_contours = params['list_sparsity_contours']
            self.list_sparsity_grid = params['list_sparsity_grid']
            self.list_sparsity_regions = params['list_sparsity_regions']
            self.list_sparsity_skels = params['list_sparsity_skels']

    def get_loaders(self, data_name, task_name, fold_name, resize_to, args, imgtype='med'):
        return get_tune_loaders(self.list_shots, 
                                self.list_sparsity_points, 
                                self.list_sparsity_contours, 
                                self.list_sparsity_grid, 
                                self.list_sparsity_regions, 
                                self.list_sparsity_skels, 
                                data_name, task_name, fold_name, resize_to, args, imgtype=imgtype)

def get_tune_loaders(shots, points, contours, grid, regions, skels, data_name, task_name, fold_name, resize_to, args, imgtype='med'):
    
    # Tuning and testing on sparsity mode 'points'.
    points_loader = []
    for n_shots in shots:
        for sparsity in points:
            
            tune_train_points_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='points', sparsity_param=sparsity, imgtype=imgtype)
            tune_train_points_loader = DataLoader(tune_train_points_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_points_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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
            
            tune_train_contours_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='contours', sparsity_param=sparsity, imgtype=imgtype)
            tune_train_contours_loader = DataLoader(tune_train_contours_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_contours_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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
            
            tune_train_grid_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='grid', sparsity_param=sparsity, imgtype=imgtype)
            tune_train_grid_loader = DataLoader(tune_train_grid_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_grid_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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
            
            tune_train_regions_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='regions', sparsity_param=sparsity, imgtype=imgtype)
            tune_train_regions_loader = DataLoader(tune_train_regions_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)

            tune_test_regions_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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
            
            tune_train_skels_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='skels', sparsity_param=sparsity, imgtype=imgtype)
            tune_train_skels_loader = DataLoader(tune_train_skels_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
            
            tune_test_skels_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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
        tune_train_dense_set = ListDataset('tune_train', data_name, task_name, fold_name, resize_to, num_shots=n_shots, sparsity_mode='dense', imgtype=imgtype)
        tune_train_dense_loader = DataLoader(tune_train_dense_set, batch_size=args['batch_size'], num_workers=args['num_workers'], shuffle=True)
        
        tune_test_dense_set = ListDataset('tune_test', data_name, task_name, fold_name, resize_to, num_shots=-1, sparsity_mode='dense', imgtype=imgtype)
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