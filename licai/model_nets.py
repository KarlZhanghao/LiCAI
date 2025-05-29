import os, glob, tqdm, tifffile
import numpy as np
import pandas as pd
from scipy import ndimage

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from licai.v22.unet3d_v22 import AttU_Net_D4
from licai.model_base import BaseModel


"""
model with AttU_Net_D4 for 3D dataset
"""
class AttUnet3D_model(BaseModel):
    """
    exp_name: save folder name
    """
    def __init__(self, exp_name="test", device="cuda", n_channels=2, n_classes=1, pos_weights=2.0):
        super().__init__(exp_name, device)
        # model
        self.model = AttU_Net_D4(n_channels=n_channels, n_classes=n_classes)
        # default loss functions   
        pos_weights = torch.full([1],pos_weights).cuda()     
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        # default optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # device        
        self.model.to(self.device) 
        self.loss_function.to(self.device)  
    
    """
    train model
    """
    def train_ds(self, trainset, batch_size=2, nepoch=300, lr=1e-4):
        # default maximum epochs
        best_score = 0
        best_loss = 1e6        
        # default scheduler
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=5, verbose=True)
        # dataloader
        train_sampler, trainidx, validx, testidx = BaseModel.split_trainset(trainset) 
        # epoch loop
        for epoch in range(0, nepoch):
            # training
            epoch_loss = self.train_epoch(trainset, train_sampler, batch_size)
            # validation
            valid_loss, eval_score = self.valid_epoch(trainset, trainidx, validx, testidx)            
            # logging
            train_score = eval_score.iloc[trainidx]
            val_score = eval_score.iloc[validx]
            test_score = eval_score.iloc[testidx]
            train_loss = train_score["loss"].mean()
            val_loss = val_score["loss"].mean()
            test_loss = test_score["loss"].mean()
            train_f1score = train_score["f1score"].mean()
            train_recall = train_score["recall"].mean()
            train_precision = train_score["precision"].mean()
            val_f1score = val_score["f1score"].mean()
            test_f1score = test_score["f1score"].mean() 
            self.logger.info(f"Epochs: {epoch}/{nepoch}.. "
                f"Epoch loss: {epoch_loss:.8f}, "
                f"Train loss: {train_loss:.8f}, "
                f"Val loss: {val_loss:.8f}, "
                f"Test loss: {test_loss:.8f}, "
                f"Train f1score: {train_f1score*100:.1f}, "
                f"Val f1score: {val_f1score*100:.1f}, "  
                f"Test f1score: {test_f1score*100:.1f}, " 
                f"Train recall: {train_recall*100:.1f}, "
                f"Train precision: {train_precision*100:.1f}, "
                f"lr: {self.optimizer.param_groups[0]['lr']*1e4:.4f}")
            # schedule lr
            scheduler.step(val_loss)
            # saving
            if best_score < val_f1score:
                best_score = val_f1score
                self.save_model(type='unet3d') 
                train_score.to_csv(os.path.join(self.exp_dir,'trainscore_bestscore.csv'), index=False)
                val_score.to_csv(os.path.join(self.exp_dir,'valscore_bestscore.csv'), index=False)
                test_score.to_csv(os.path.join(self.exp_dir,'testscore_bestscore.csv'), index=False)
            if best_loss > test_loss:
                best_loss = test_loss
                self.save_model(type='unet3d') 
                train_score.to_csv(os.path.join(self.exp_dir,'trainscore_bestloss.csv'), index=False)
                val_score.to_csv(os.path.join(self.exp_dir,'valscore_bestloss.csv'), index=False)
                test_score.to_csv(os.path.join(self.exp_dir,'testscore_bestloss.csv'), index=False)
            # stopping criteria
            if self.optimizer.param_groups[0]['lr'] < 1e-6:
                break                
        return self.logger

    """
    one training epoch
    """
    def train_epoch(self, ds, train_sampler, batch_size):
        # dataloader
        train_loader = DataLoader(ds, batch_size=batch_size, pin_memory=True, num_workers=batch_size, sampler=train_sampler)
        # training
        self.model.train()
        epoch_loss = 0
        # disp training progress
        batch_tqdm = tqdm.tqdm(train_loader, total=len(train_loader))
        for batch in batch_tqdm:
            low, gt = batch['image'], batch['mask']            
            low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
            gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)
            pred = self.model(low)
            loss = self.loss_function(pred, gt)
            epoch_loss += loss.item()
            # backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            # tqdm
            batch_tqdm.set_description(f"Train")
            batch_tqdm.set_postfix(loss=loss.item())
        epoch_loss = epoch_loss/len(train_loader)
        return epoch_loss
        
    """
    one validation epoch
    """
    def valid_epoch(self, ds):
        # dataloader
        n_tform, is_tform = ds.n_tform, ds.is_tform
        ds.n_tform=1
        ds.is_tform=False
        val_loader = DataLoader(ds, batch_size=1, pin_memory=True)
        # eval mode
        self.model.eval()
        valid_loss = 0
        # disp training progress
        batch_tqdm = tqdm.tqdm(val_loader, total=len(val_loader))
        # record scores
        scoredict = {'dataname':[],'loss':[],'acc':[],'moc':[],'f1score':[],'recall':[],'precision':[],'ntp':[],'nfp':[],'ntn':[],'tfn':[]}
        scoretable = pd.DataFrame(scoredict)
        scoretable = scoretable.T
        with torch.no_grad():
            for batch in batch_tqdm:                
                #
                low, gt =  batch['image'], batch['mask']
                low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
                gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)
                # prediction with cropped patches
                pred = self.pred_crop(low, self.model, self.device)
                loss = self.loss_function(pred, gt)
                valid_loss += loss.data.item()
                # cal scores
                pred = torch.sigmoid(pred) > 0.5
                score = self.cal_scores(pred[0,0,:,:,:].cpu(), gt[0,0,:,:,:].cpu())
                tabletmp = pd.Series(score)
                tabletmp['dataname'] = batch['filename'][0]
                tabletmp['loss'] = loss.item()
                scoretable = pd.concat([scoretable, tabletmp], axis=1)
                # tqdm
                batch_tqdm.set_description(f"Valid")                
                batch_tqdm.set_postfix(loss=loss.item())
            valid_loss = valid_loss/len(val_loader)
        # reset dataset
        ds.n_tform, ds.is_tform = n_tform, is_tform
        return valid_loss, scoretable.T
    
    """
    inference dataset, save images and scores
    """
    def eval_ds(self, ds, saveinfo):
        issave = saveinfo['issave']
        save_name = saveinfo['save_name']
        # dataloader
        n_tform, is_tform = ds.n_tform, ds.is_tform
        ds.n_tform=1
        ds.is_tform=False
        eval_loader = DataLoader(ds, batch_size=1, pin_memory=True)
        # eval mode
        self.model.eval()
        # disp training progress
        batch_tqdm = tqdm.tqdm(eval_loader, total=len(eval_loader))
        # record scores
        scoredict = {'dataname':[],'acc':[],'moc':[],'f1score':[],'recall':[],'precision':[],'ntp':[],'nfp':[],'ntn':[],'tfn':[]}
        scoretable = pd.DataFrame(scoredict)
        scoretable = scoretable.T
        with torch.no_grad():
            for batch in batch_tqdm:                
                #
                low, gt =  batch['image'], batch['mask']
                low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
                gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)
                # prediction with cropped patches
                pred = self.pred_crop(low, self.model, self.device)
                # cal scores
                pred = torch.sigmoid(pred) > 0.5
                score = self.cal_scores(pred[0,0,:,:,:].cpu(), gt[0,0,:,:,:].cpu())
                tabletmp = pd.Series(score)
                tabletmp['dataname'] = batch['filename'][0]
                scoretable = pd.concat([scoretable, tabletmp], axis=1)
                # tqdm
                batch_tqdm.set_description(f"Valid")
                # save images
                if issave:
                    predsave = pred[0,:,:,:,:].cpu().numpy()
                    predsave = np.squeeze(predsave, axis=0)
                    xyzoom = np.float32(batch['rawnp'])/np.float32(predsave.shape[-1])
                    zoom=[np.float32(batch['rawnz'][0])/predsave.shape[0],xyzoom[0],xyzoom[0]]
                    predsave = ndimage.zoom(predsave,zoom=zoom,order=0,mode='mirror',prefilter=False)>0.5
                    tifffile.imsave(os.path.join(batch['data_dir'][0],batch['filename'][0],save_name+'.tif'), predsave.astype(np.uint8)*255)
        # reset dataset
        ds.n_tform, ds.is_tform = n_tform, is_tform
        return scoretable.T

    def finetune(self, trainset, evalset, batch_size=16, nepoch=300, lr=1e-4):
        # default maximum epochs
        bestmapping = 0
        # default scheduler
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=5, verbose=True)
        # dataloader
        train_sampler, val_sampler,_,_ = BaseModel.split_trainset(trainset) 
        train_loader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, num_workers=batch_size, sampler=train_sampler)              
        # training
        for epoch in range(0, nepoch):
            self.model.train()
            #batch_tqdm = tqdm.tqdm(train_loader, total=len(train_loader))
            #for batch in batch_tqdm:
            for batch in train_loader:
                low, gt = batch[0], batch[1]            
                low = low.to(device=self.device, dtype=torch.float32, non_blocking=True)
                gt = gt.to(device=self.device, dtype=torch.float32, non_blocking=True)
                pred = self.model(low)
                loss = self.loss_function(pred, gt)
                mapping = self.eval_dataset(evalset)
                loss_all = loss
                loss_all.item += loss.item() + mapping
                # backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                # tqdm
                #batch_tqdm.set_description(f"Train: ")
                #batch_tqdm.set_postfix(loss=loss.item())
                epoch_loss = epoch_loss/len(train_loader)
                # logging and saving
                self.logger.info('Epochs: [%d/%d]; Train loss: %0.6f; Valid loss: %0.6f; mapping: %.6f; lr: %.6f \n'% (epoch, nepoch, epoch_loss, valid_loss, mapping, self.optimizer.param_groups[0]['lr']))
                if mapping > bestmapping:
                    bestmapping = mapping
                    self.save_model() 
                if self.optimizer.param_groups[0]['lr']<1e-6:
                    break
        return self.logger
        
    def pred_folder(self, data_dir, target_dir):
        cycs = [folder for folder in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir,folder))]
        for cyc in cycs:
            img_path = os.path.join(data_dir, cyc)
            imgfiles = glob.glob(os.path.join(img_path,"*.tif"))        
            for imgfile in imgfiles:
                img = tifffile.imread(imgfile)               
                img = self.transforms_predict(img) 
                with torch.no_grad():
                    pred = self.model(img.to(self.device))
                savepath = os.path.join(target_dir,cyc)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                tifffile.imsave(os.path.join(savepath,os.path.basename(imgfile)),np.int16(pred.cpu().numpy()*5000))
            