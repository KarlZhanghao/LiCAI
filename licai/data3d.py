from licai.data_base import BaseDataset

import os
import numpy as np
from scipy import ndimage
import tifffile
import torch

"""
3D dataset for ratio image
所有的Data类最终输出数据统一为 TCZYX
输入数据不再区分2D和3D
只判定需要读取哪些数据,分别处理后放在C通道
后续网络训练和预测时根据实际情况做变换
"""
class RatioDataset(BaseDataset):
    """
    images are resized to 100 nm/pixel, and 24 z slices
    """
    def __init__(self, 
        data_dir, 
        n_tform=1, 
        is_tform=True,        
        nz=24,
        scale=0.668, 
        fluogain=1.0,  
        fluoimg='fluo.tif', 
        ratioimg='ratio.tif', 
        maskimg='label.tif',
        norm='percentile'):
        
        super().__init__(data_dir, n_tform, is_tform)
        # image files
        self.imgnames = (fluoimg,ratioimg,maskimg)
        # image size
        self.shape = 256
        self.scale = scale
        self.fluogain = fluogain
        self.nz = nz
        self.norm = norm
    
    """
    read images and normalize
    """
    def readimg(self, idx):
        imgs = ()
        for files in self.imgnames:
            img = tifffile.imread(os.path.join(self.data_dir,idx,files))            
            # image type
            if img.dtype == 'uint16':
                img = (img/65535.)
            elif img.dtype == 'uint8':
                img = (img/255.)
            # foreground intensity normalization to std=0.1
            if files == self.imgnames[0]:
                if len(img.shape) == 3: #ZYX
                    Nt = 1
                    Nz = img.shape[0]
                    inp = img.shape[1] 
                elif len(img.shape) == 4: #TZXY
                    Nt = img.shape[0]
                    Nz = img.shape[1]
                    inp = img.shape[2]
            # reshape images with TZXY
            img = img.reshape((Nt, Nz, inp, inp))
            # append images to save             
            imgs = imgs+(img,)             
        return imgs+(Nt, Nz,inp,)
    
    """
    transformation for data augmentation
    """
    def my_transforms(self, img, yy, xx, angle, zoomratio):
        if self.is_tform:
            # axial reslice, xy zoom, rotation, crop
            img_nd = ndimage.zoom(img,zoom=[self.nz/img.shape[0],self.scale*zoomratio,self.scale*zoomratio],order=0,mode='mirror',prefilter=False)
            img_nd = ndimage.rotate(img_nd,angle,axes=(1,2),reshape=False,mode='mirror',prefilter=False)
            img_nd = img_nd[:,yy:yy+self.shape,xx:xx+self.shape]
        else:
            img_nd = ndimage.zoom(img,zoom=[self.nz/img.shape[0],self.scale,self.scale],order=0,mode='mirror',prefilter=False)
        # ZYX to CZYX
        img_nd = np.expand_dims(img_nd, axis=0)
        return img_nd

    """
    get item
    """
    def __getitem__(self, i):    
        # read images
        idx = self.ids[i%self.ndata] 
        (img,ratio,mask,Nt, Nz,inp) = self.readimg(idx)  
        # norm fluo image
        for tt in range(Nt):#TZYX
            mynorm = self.normalize_fluo_percentile if self.norm=='percentile' else self.normalize_fluo_std
            img[tt,:,:,:] = mynorm(img[tt,:,:,:], gain=self.fluogain)
        # cal transformation parameters
        dx = self.shape
        if self.is_tform:   
            intratio = np.random.uniform(0.8,1.2)
            zoomratio = np.random.uniform(0.9,1.1)
            angle = np.random.randint(-90,90)
            yy = np.random.randint(np.round(inp*self.scale*zoomratio*0.15),np.round(inp*self.scale*zoomratio*0.85)-dx-1)
            xx = np.random.randint(np.round(inp*self.scale*zoomratio*0.15),np.round(inp*self.scale*zoomratio*0.85)-dx-1)     
        else:
            intratio = 1.0
            zoomratio = 1
            angle = 0
            yy=0            
            xx=0
        # transform images
        for index, imgtmp in enumerate([img, ratio, mask]):
            for tt in range(Nt):#TZYX
                img_ttmp = self.my_transforms(imgtmp[tt,:,:,:],yy,xx,angle,zoomratio) #CZYX       
                img_ttmp = np.expand_dims(img_ttmp, axis=0) #TCZYX
                if tt == 0:
                    img_t = img_ttmp
                else:
                    img_t = np.concatenate((img_t,img_ttmp),axis=0)
            if index == 0:
                img_tall = [img_t,]
            else:
                img_tall.append(img_t)
        # concatenate intensity and ratio images, TCZYX
        img = img_tall[0]
        ratio = img_tall[1]
        mask = img_tall[2]
        img = np.concatenate((img*intratio, ratio), axis=1)       
        return {"image": torch.from_numpy(img.astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.float32)),
                "data_dir": self.data_dir,
                "rawnt":Nt,
                "rawnp":inp,
                "rawnz":Nz,
                "filename": idx}