import os
import numpy as np
from torch.utils.data import Dataset

"""
Base class for all datasets, read datasets from folders
"""
class BaseDataset(Dataset):
    
    """
    is_tform: whether to apply data augmentation, True during model training
    n_tform: number of data augmentation
    """
    def __init__(self, 
        data_dir, 
        n_tform=1, 
        is_tform=True):
        # data structure  
        if n_tform == 0:
            self.ids = [data_dir]
        else:        
            self.ids = [os.path.join(data_dir,folder) for folder in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir,folder))]
        self.ndata = len(self.ids)
        # data augm
        self.n_tform = np.round(n_tform)
        if self.n_tform > 1:
            self.is_tform = True
        else:
            self.is_tform = is_tform        
    
    """
    number of data          
    """
    def __len__(self):
        if self.n_tform == 0:
            nlen = 1
        else:
            nlen = self.n_tform*self.ndata
        return nlen
    
    def __getitem__(self,i):
        return NotImplemented
    
    """
    concatenate two datasets
    """
    def append(self, ds):
        self.ids = self.ids + ds.ids
        self.ndata = len(self.ids)

    """
    normalize intensity images
    """
    @staticmethod
    def normalize_fluo(img, gain):
        fg = img[img>50./65535.]
        img = img/fg.std()*0.1*gain
        return img
        