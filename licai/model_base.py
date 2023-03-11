import os, logging
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

"""
Base class for all models
save info, logging, model save, dataset split, etc.
"""
class BaseModel():
    """
    Model dir, device
    """
    def __init__(self, exp_name="test", device="cuda"):
        self.exp_name = exp_name
        self.exp_dir = "./exp/"+exp_name+"/"
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.logger = self.get_logger()
        #
        self.device = torch.device(device)       
    
        """
    setup logger
    """
    def get_logger(self):
        logger = logging.getLogger()     
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.exp_dir, "log.txt"))
        sh = logging.StreamHandler()
        fa = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(fa)
        sh.setFormatter(fa)
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(sh)
        return logger
    
    """
    initialize model parameters
    """
    @staticmethod
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)


    """
    load model
    """
    def load(self, model_path=None, type='Undefined'):
        if model_path is None:
            checkpoint = torch.load(os.path.join(self.exp_dir, "bestloss.pth"), map_location=self.device)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)  
        if checkpoint['type'] != type:
            raise Exception('incorret model type')  
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("model loaded")
        
    """
    save model
    """
    def save_model(self, save_dir=None, type=None):
        checkpoint = {
            'type':type,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        if save_dir is None:
            save_dir = self.exp_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        savepath = os.path.join(save_dir,'bestloss.pth')
        torch.save(checkpoint,savepath)

    """
    trans model
    """
    def trans_model(self, load_dir=None, save_dir=None, type=None):
        if load_dir is None:
            load_dir = self.exp_dir
        checkpoint = torch.load(os.path.join(load_dir, "bestloss.pth"), map_location=self.device)
        checkpoint = {
            'type':type,
            'model': checkpoint
            }
        if save_dir is None:
            save_dir = self.exp_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)        
        savepath = os.path.join(save_dir,'bestloss.pth')
        torch.save(checkpoint,savepath)
    
    """
    split trainset into trainset, validation, and test set
    """
    @staticmethod
    def split_trainset(trainset):
        datasize = int(trainset.ndata)
        nt = trainset.n_tform
        trainsize = int(0.6*datasize)
        valsize = int(0.8*datasize)
        indices = list(range(datasize))
        np.random.seed(42)
        np.random.shuffle(indices)
        trainidx, validx, testidx = indices[:trainsize], indices[trainsize:valsize], indices[valsize:]
        trainsampler = SubsetRandomSampler(np.tile(trainidx,nt))
        return trainsampler, trainidx, validx, testidx

    """
    crop image for prediction, in case of cuda OOM error
    """    
    @staticmethod
    def pred_crop(low, model, device):
        # cal crop params
        inp=low.shape[-1]
        np_crop = 256
        dp_crop0 = 32
        N = np.int32(np.ceil((inp-np_crop)/(np_crop-dp_crop0))+1)
        dp_crop = np.int32(np_crop-np.round((inp-np_crop)/(N-1)))
        cp1 = np.arange(0, (np_crop-dp_crop)*N-1, np_crop-dp_crop)
        cp1[-1]=inp-np_crop-1
        cp2 = cp1+np_crop
        dp_crop=cp2[0]-cp1[1]
        #
        pred = torch.zeros([1,1,low.shape[2],low.shape[3],low.shape[4]],dtype=low.dtype,device=device)
        with torch.no_grad():
            for crow in range(N):
                for ccol in range(N):                    
                    lowcrop = low[:,:,:,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]
                    predcrop = model(lowcrop.to(device))
                    pred[:,:,:,cp1[crow]:cp2[crow],cp1[ccol]:cp2[ccol]]=predcrop
        return pred
    
        """
    calculate the scores of prediction
    """
    @staticmethod
    def cal_scores(pred, gt):
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
        return {'acc':accuracy,
            'moc':moc.numpy(),
            'f1score':f1score,
            'recall':recall,
            'precision':precision,
            'ntp':TP,
            'nfp':FP,
            'ntn':TN,
            'tfn':FN}