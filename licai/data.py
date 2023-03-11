from os.path import splitext
from os import listdir

import numpy as np
from scipy import ndimage
import random
import tifffile

import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as TF

class Rand3dRatioDataset(Dataset):
    """"""
    def __init__(self, data_dir, fluoimg='fluo.tif', ratioimg='ratio.tif', maskimg='mask.tif'):
        # read data folders
        self.data_dir = data_dir        
        self.ids = [file for file in listdir(self.data_dir)
                    if not file.startswith('.')]                    
        self.ndata = len(self.ids)
        # image files
        self.fluoimg = '/' + fluoimg
        self.ratioimg = '/' + ratioimg
        self.maskimg = '/' + maskimg        
        
        
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def my_transforms(cls, img):
        # normalization
        if img.dtype == 'uint16':
            img = img/65535.
        elif img.dtype == 'uint8':
            img = img/255.
        
        if img.shape[1] == 1200:
            img = ndimage.zoom(img,zoom=[20./img.shape[0],802./1200,802./1200],order=0,mode='nearest',prefilter=False)
            img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
            img,np.expand_dims(img[-1,:,:],axis=0),np.expand_dims(img[-1,:,:],axis=0)),axis=0)
        elif img.shape[1] == 944:
            img = ndimage.zoom(img,zoom=[20./img.shape[0],648./944,648./944],order=0,mode='nearest',prefilter=False)
            img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
            img,np.expand_dims(img[-1,:,:],axis=0),np.expand_dims(img[-1,:,:],axis=0)),axis=0)
        
        img_nd = img        
        # ZHW to CZHW
        img_nd = np.expand_dims(img_nd, axis=0)      
        
        return img_nd

    def __getitem__(self, i):
        # data information
        idx = self.ids[i%self.ndata]
        tmpstr = idx.split('_NZ_')
        nz = int(tmpstr[1])
        # read images
        img = tifffile.imread(self.data_dir + idx + self.fluoimg) 
        ratio = tifffile.imread(self.data_dir + idx + self.ratioimg) 
        mask = tifffile.imread(self.data_dir + idx + self.maskimg)
        inp = img.shape[1]
        # transformation parameters
        intratio = 2.0
        # trainsformation
        img = self.my_transforms(img)
        ratio = self.my_transforms(ratio)
        img = np.concatenate((np.minimum(img*intratio,1), ratio), axis=0)
        mask = self.my_transforms(mask)
        # print(idx)
        return {"image": torch.from_numpy(img),
                "mask": torch.from_numpy(mask.astype(np.float32)),
                "filename": idx,
                "nz": nz,
                "np": inp}

class Rand3dVolDataset(Dataset):
    """"""
    def __init__(self, data_dir, fluoimg='fluo.tif',maskimg='mask.tif'):
        # read data folders
        self.data_dir = data_dir        
        self.ids = [file for file in listdir(self.data_dir)
                    if not file.startswith('.')]                    
        self.ndata = len(self.ids)
        # image files
        self.fluoimg = '/' + fluoimg
        self.maskimg = '/' + maskimg        
        
    def __len__(self):
        return len(self.ids)
    
    @classmethod
    def my_transforms(cls, img):             
        if img.shape[1] == 1200:
            img = ndimage.zoom(img,zoom=[20./img.shape[0],512./1200,512./1200],order=0,mode='nearest',prefilter=False)
            img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
            img,np.expand_dims(img[-1,:,:],axis=0),np.expand_dims(img[-1,:,:],axis=0)),axis=0)   
        elif img.shape[1] == 944:
            img = ndimage.zoom(img,zoom=[20./img.shape[0],402./944,402./944],order=0,mode='nearest',prefilter=False)
            img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
            img,np.expand_dims(img[-1,:,:],axis=0),np.expand_dims(img[-1,:,:],axis=0)),axis=0)             
            
        img_nd = img        
        # ZHW to CZHW
        img_nd = np.expand_dims(img_nd, axis=0)      
        
        return img_nd

    def __getitem__(self, i):
        # data information
        idx = self.ids[i%self.ndata]
        tmpstr = idx.split('_NZ_')
        nz = int(tmpstr[1])
        # read images
        img = tifffile.imread(self.data_dir + idx + self.fluoimg) 
        mask = tifffile.imread(self.data_dir + idx + self.maskimg)
        inp = img.shape[1]   
        # trainsformation
        img = self.my_transforms(img)/255.
        mask = self.my_transforms(mask)
        #
        return {"image": torch.from_numpy(img),
                "mask": torch.from_numpy(mask.astype(np.float32)),
                "filename": idx,
                "nz": nz,
                "np": inp}