from licai.data import Rand3dRatioDataset, Rand3dVolDataset
from licai.net import AttU_Net3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from scipy import ndimage
import tifffile
import pandas as pd
from tqdm import tqdm
import os

def transforms_post(hr, lr, type='2D'):  
    if type == '2D':  
        hr=hr.reshape((hr.shape[0]*hr.shape[1],)+hr.shape[2:])
        lr=lr.reshape((lr.shape[0]*lr.shape[1],)+lr.shape[2:])
    return hr.to(torch.float32), lr.to(torch.float32)

"""general prediction function with crop
!!! All images are in the format of TCZYX
"""
def pred_crop(low, pred, model, device, type='3D'):    
    # cal crop params
    inp=low.shape[-1]
    np_crop = 256
    dp_crop0 = 32
    dp_crop1 = np.int32(np.round(dp_crop0/2))
    N = np.int32(np.ceil((inp-np_crop)/(np_crop-dp_crop0))+1)
    dp_crop = np.int32(np_crop-np.round((inp-np_crop)/(N-1)))
    cp1 = np.arange(0, (np_crop-dp_crop)*N-1, np_crop-dp_crop)
    cp1[-1]=inp-np_crop-1
    cp2 = cp1+np_crop
    dp_crop=cp2[0]-cp1[1]
    # low in TCZYX
    with torch.no_grad():
        for bb in range(low.shape[0]):
            for crow in range(N):
                for ccol in range(N):   
                    if type == '3D':                 
                        lowcrop = low[bb:bb+1,:,:,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]
                        predcrop = model(lowcrop.to(device))
                    else:
                        lowcrop = low[bb:bb+1,:,:,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]
                        lowcrop = lowcrop.permute(0,2,1,3,4).squeeze(0) #ZCYX / BCYX
                        predcrop = model(lowcrop.to(device))
                        predcrop = predcrop.unsqueeze(0).permute(0,2,1,3,4) #BCZYX
                    # modified montage
                    cpr1=cp1[crow]+dp_crop1 if crow>0 else cp1[crow]
                    cpr2=dp_crop1 if crow>0 else 0
                    cpc1=cp1[ccol]+dp_crop1 if ccol>0 else cp1[ccol]
                    cpc2=dp_crop1 if ccol>0 else 0 
                    pred[bb,:,:,cpr1:cp2[crow],cpc1:cp2[ccol]]=predcrop[0,:,:,cpr2:,cpc2:]
    return pred

def pred_binary(predset, net, save_info, type="2D", lossfun=nn.BCEWithLogitsLoss().cpu()):
    issave = save_info['issave']
    save_name = save_info["save_name"]
    # pred and eval data
    predloader = DataLoader(predset, batch_size=1, pin_memory=True)
    n_val = len(predloader)  
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.eval()
    with torch.no_grad():
        with tqdm(total=n_val, desc='round', unit='batch', leave=False) as pbar:
            scoredict = {'dataname':[],'acc':[],'moc':[],'f1score':[],'recall':[],'precision':[],'ntp':[],'nfp':[],'ntn':[],'tfn':[]}
            scoretable = pd.DataFrame(scoredict)
            scoretable = scoretable.T
            for batch in predloader:
                ## read data
                imgs, masks, Nt, Nz, Np = batch['image'], batch['mask'], batch['rawnt'], batch['rawnz'], batch['rawnp']
                #Nc0, Nc1 = batch['rawnc0'], batch['rawnc1']   
                # BTCZYX to TCZYX
                imgs = imgs.reshape((imgs.shape[0]*imgs.shape[1],)+imgs.shape[2:])
                masks = masks.reshape((masks.shape[0]*masks.shape[1],)+masks.shape[2:])
                pred = torch.zeros(masks.shape,dtype=masks.dtype, device=device)             
                pred = pred_crop(imgs, pred, net, device, type=type) # imgs: BTCZYX              
                # cal loss
                loss = 0
                for tt in range(pred.shape[0]):
                    loss+=lossfun(pred[tt,:],masks[tt,:].to(device))
                loss = loss/pred.shape[0]
                # evaluate performance                                           
                pred = torch.sigmoid(pred) > 0.5
                score = eval_data(pred.cpu(), masks.cpu())
                tabletmp = pd.Series(score)
                tabletmp['dataname'] = batch['filename'][0]
                tabletmp['loss'] = loss.item()                
                scoretable = pd.concat([scoretable, tabletmp], axis=1)
                # save image
                if issave:
                    predsave = pred.cpu().numpy()
                    predsave = np.squeeze(predsave, axis=1)
                    xyzoom = np.float32(batch['rawnp'])/np.float32(predsave.shape[-1])
                    zoom=[np.float32(batch['rawnz'][0])/predsave.shape[-3],xyzoom[0],xyzoom[0]]
                    for tt in range(predsave.shape[0]):
                        predsave_tmp = ndimage.zoom(predsave[tt,:,:,:],zoom=zoom,order=0,mode='mirror',prefilter=False)
                        predsave_tmp = np.expand_dims(predsave_tmp, axis=0)
                        if tt == 0:
                            predsave_tform = predsave_tmp
                        else:
                            predsave_tform = np.concatenate((predsave_tform,predsave_tmp),axis=0)
                    tifffile.imsave(os.path.join(batch['data_dir'][0],batch['filename'][0],save_name+'.tif'), 
                                    predsave_tform.astype(np.uint8)*255, 
                                    imagej=True, metadata={'axes': 'TZYX'})
                # updata bar    
                pbar.update()
    
    scoretable = scoretable.T
    scoretable.to_csv(os.path.join(predset.data_dir,save_name+'_scoretable.csv'),index=False)
    return scoretable

def predfolder_ratio3d_binary(pred_dir, model_info, save_info):
    #
    model_file = model_info["model_file"]
    nclasses = model_info["nclasses"]
    fluoimg = model_info["fluoimg"]
    ratioimg = model_info["ratioimg"]
    maskimg = model_info["maskimg"]
    #
    issave = save_info['issave']
    save_name = save_info["save_name"]
    # load data
    predset = Rand3dRatioDataset(pred_dir,fluoimg=fluoimg,ratioimg=ratioimg,maskimg=maskimg)
    batch_size = 1
    pred_loader = DataLoader(predset, batch_size=batch_size, pin_memory=True)
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AttU_Net3D(n_channels=2, n_classes=nclasses)
    state_dict = torch.load(model_file, map_location=device)
    net.load_state_dict(state_dict)
    net.to(device=device)
    net.eval()
    # cal crop information
    dataiter = iter(pred_loader)
    nextdata = dataiter.next()
    inp = nextdata['image'].shape[4]
    crop_info = cal_crop_param(inp)
    c0 = crop_info['c0']
    c1 = crop_info['c1']
    ncrop = len(c0)
    mcrop = crop_info['mod']    
    medge = crop_info['edge']
    medge = np.expand_dims(np.expand_dims(medge,axis=0),axis=0)
    if nclasses > 1:
        mcrop = np.tile(np.expand_dims(mcrop,axis=0),(nclasses,1,1))
        medge = np.tile(medge, (1,nclasses,1,1,1))
    # pred and eval data
    n_val = len(pred_loader)  
    mask_type = torch.float32 if net.n_classes == 1 else torch.long    
    with torch.no_grad():
        with tqdm(total=n_val, desc='round', unit='batch', leave=False) as pbar:
            if nclasses == 1:
                scoredict = {'dataname':[],'acc':[],'moc':[],'f1score':[],'recall':[],'precision':[],'ntp':[],'nfp':[],'ntn':[],'tfn':[]}
                scoretable = pd.DataFrame(scoredict)
            for batch in pred_loader:
                ## read data
                imgs, masks = batch['image'], batch['mask']
                pred = torch.zeros_like(masks)
                # crop and pred data
                for crow in range(ncrop):
                    for ccol in range(ncrop):
                        imgcrop = imgs[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]]
                        if nclasses == 1:
                            output = net.forward(imgcrop.to(device=device, dtype=torch.float32))                        
                            predcrop = torch.sigmoid(output.cpu())
                        else:
                            output = net.forward(imgcrop.to(device=device, dtype=torch.long))                        
                            activation = nn.Softmax(dim=1)
                            predcrop = activation(output.cpu())
                            
                        pred[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]] = pred[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]] + predcrop*mcrop
                # merge data
                pred = pred/medge                
                # evaluate performance
                if nclasses == 1:
                    pred = pred > 0.5
                    score = eval_data(pred[0,0,2:-2,:,:], masks[0,0,2:-2,:,:])
                    tabletmp = pd.Series(score)
                    tabletmp['dataname'] = batch['filename'][0]
                    scoretable = scoretable.append(tabletmp, ignore_index=True)
                # save image
                if issave:
                    predsave = pred[0,:,2:-2,:,:].numpy()
                    predsave = np.squeeze(predsave, axis=0)
                    xyzoom = np.float32(batch['np'])/np.float32(predsave.shape[-1])
                    zoom=[np.float32(batch['nz'][0])/20.,xyzoom[0],xyzoom[0]]
                    predsave = ndimage.zoom(predsave,zoom=zoom,order=0,mode='nearest',prefilter=False)
                    tifffile.imsave(pred_dir+batch['filename'][0]+ '/'+save_name+'.tif', np.uint8(predsave*255))
                # updata bar    
                pbar.update()
    #if issave:
    #    scoretable.to_csv(pred_dir+batch['filename'][0]+'/scoretable.csv',index=False)
        
    return scoretable

def predfolder_int3d_multi(pred_dir, model_info, save_info):
    #
    model_file = model_info["model_file"]
    nclasses = model_info["nclasses"]
    fluoimg = model_info["fluoimg"]
    maskimg = model_info["maskimg"]
    #
    issave = save_info['issave']
    save_name = save_info["save_name"]
    # load data
    predset = Rand3dVolDataset(pred_dir,fluoimg=fluoimg,maskimg=maskimg)
    batch_size = 1
    pred_loader = DataLoader(predset, batch_size=batch_size, pin_memory=True)
    # load model
    net = AttU_Net3D(n_channels=1, n_classes=nclasses)
    state_dict = torch.load(model_file)
    net.load_state_dict(state_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.eval()
    # pred and eval data
    n_val = len(pred_loader)  
    mask_type = torch.float32 if net.n_classes == 1 else torch.long    
    with torch.no_grad():
        with tqdm(total=n_val, desc='round', unit='batch', leave=False) as pbar:
            scoredict = {'dataname':[],'acc':[],'moc':[],'f1score':[],'recall':[],'precision':[],'ntp':[],'nfp':[],'ntn':[],'tfn':[]}
            scoretable = pd.DataFrame(scoredict)
            for batch in pred_loader:
                ## read data
                imgs, masks = batch['image'], batch['mask']
                pred = torch.zeros([1,nclasses,imgs.shape[2],imgs.shape[3],imgs.shape[4]],dtype=torch.float)
                #
                inp = imgs.shape[4]
                crop_info = cal_crop_param(inp)
                c0 = crop_info['c0']
                c1 = crop_info['c1']
                ncrop = len(c0)
                mcrop = crop_info['mod']    
                medge = crop_info['edge']
                medge = np.expand_dims(np.expand_dims(medge,axis=0),axis=0)
                mcrop = np.tile(np.expand_dims(mcrop,axis=0),(nclasses,1,1,1))
                medge = np.tile(medge, (1,nclasses,1,1,1))
                # crop and pred data
                for crow in range(ncrop):
                    for ccol in range(ncrop):
                        imgcrop = imgs[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]]
                        output = net.forward(imgcrop.to(device=device, dtype=torch.float32))                        
                        activation = nn.Softmax(dim=1)
                        predcrop = activation(output.cpu())                            
                        pred[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]] = pred[:,:,:,c0[crow]:c1[crow],c0[ccol]:c1[ccol]] + predcrop*mcrop
                # merge data
                pred = pred/medge                
                # evaluate performance
                if nclasses>1:
                    pred = pred.argmax(dim=1)
                    predsave = pred[0,2:-2,:,:]
                    masksave = masks[0,0,2:-2,:,:]
                    for cc in range(nclasses):
                        thispred = (predsave == cc)
                        thisgt = (masksave == cc)
                        score = eval_data(thispred, thisgt)
                        tabletmp = pd.Series(score)
                        tabletmp['dataname'] = batch['filename'][0]+'_'+str(cc)
                        scoretable = scoretable.append(tabletmp, ignore_index=True)
                # save image                
                if issave:
                    predsave = predsave.numpy()
                    xyzoom = np.float32(batch['np'])/np.float32(predsave.shape[-1])
                    zoom=[np.float32(batch['nz'][0])/20.,xyzoom[0],xyzoom[0]]
                    predsave = ndimage.zoom(predsave,zoom=zoom,order=0,mode='nearest',prefilter=False)
                    for cc in range(nclasses):
                        thispredsave = predsave==cc
                        tifffile.imsave(pred_dir+batch['filename'][0]+ '/'+save_name+'_'+str(cc)+'.tif', np.uint8(thispredsave*255))
                    tifffile.imsave(pred_dir+batch['filename'][0]+ '/'+save_name+'.tif', np.uint8(predsave))
                # updata bar    
                pbar.update()
    #if issave:
    #    scoretable.to_csv(pred_dir+batch['filename'][0]+'/scoretable.csv',index=False)
        
    return scoretable


def eval_data(pred, gt):
    FP = ((pred == 1).numpy()&(gt == 0).numpy()).sum()
    FN = ((pred == 0).numpy()&(gt == 1).numpy()).sum()
    TP = ((pred == 1).numpy()&(gt == 1).numpy()).sum()
    TN = ((pred == 0).numpy()&(gt == 0).numpy()).sum()            
    Ntot = FP + FN + TP + TN + 1
    # cal score    
    accuracy = (TP+TN)/Ntot
    precision = TP/(TP+FP+1)
    recall = TP/(TP+FN+1)
    f1score = 2*precision*recall/(precision+recall+1e-6)
    # cal moc
    tmp1 = gt*pred
    tmp2 = gt*gt
    tmp3 = pred*pred
    moc = tmp1.sum()/np.sqrt(tmp2.sum()*tmp3.sum())
    #
    return {'acc':accuracy,'moc':moc.numpy(),'f1score':f1score,'recall':recall,'precision':precision,'ntp':TP,'nfp':FP,'ntn':TN,'tfn':FN}

def cal_crop_param(inp):
    if inp == 802:
        np_crop = 256
        dp_crop = 182
        N = 4
    elif inp == 648:
        np_crop = 256
        dp_crop = 196
        N = 3
    elif inp == 512:
        np_crop = 256
        dp_crop = 128
        N = 3
    elif inp == 402:
        np_crop = 256
        dp_crop = 146
        N = 2
    # cal crop indices
    cp1 = np.arange(0, dp_crop*N-1, dp_crop)
    cp2 = cp1+np_crop
    # cal merge coef
    m1 = np.ones((1,np_crop))
    m1[0,cp1[1]:cp2[0]] = np.linspace(1,0,np_crop-dp_crop)
    m2 = np.ones((1,np_crop))
    m2[0,0:(np_crop-dp_crop)] = np.linspace(0,1,np_crop-dp_crop)
    # mod coef
    mright = np.tile(np.expand_dims(m1,axis=0), (24, np_crop, 1))
    mleft = np.tile(np.expand_dims(m2,axis=0), (24, np_crop, 1))
    mlow = np.tile(np.expand_dims(m1,axis=0).transpose(0,2,1), (24, 1, np_crop))
    mup = np.tile(np.expand_dims(m2,axis=0).transpose(0,2,1), (24, 1, np_crop))
    mod = mright*mleft*mlow*mup
    # edge coef
    edge1 = np.ones((24,inp,inp))
    edge2 = np.ones((24,inp,inp))
    edge1[:,:,0:np_crop] = np.tile(np.expand_dims(m2,axis=0),(24,inp,1))
    edge1[:,:,-np_crop:] = np.tile(np.expand_dims(m1,axis=0),(24,inp,1))
    edge2[:,0:np_crop,:] = np.tile(np.expand_dims(m2,axis=0).transpose(0,2,1),(24,1,inp))
    edge2[:,-np_crop:,:] = np.tile(np.expand_dims(m1,axis=0).transpose(0,2,1),(24,1,inp))
    edge = edge1*edge2+1e-6;
    #
    return {"Np":inp, "c0":cp1, "c1": cp2, "m0": m1, "m1":m2, "mod":mod, "edge":edge}
    
def transform_img3d(img):
    if img.shape[1] == 1200:
        img = ndimage.zoom(img,zoom=[20./img.shape[0],802./1200,802./1200],order=0,mode='nearest',prefilter=False)
        img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
        img,np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0)),axis=0)
    elif img.shape[1] == 944:
        img = ndimage.zoom(img,zoom=[20./img.shape[0],648./944,648./944],order=0,mode='nearest',prefilter=False)
        img = np.concatenate((np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0),
        img,np.expand_dims(img[0,:,:],axis=0),np.expand_dims(img[0,:,:],axis=0)),axis=0)
    #
    img_nd = np.expand_dims(img, axis=0)   
    return img_nd