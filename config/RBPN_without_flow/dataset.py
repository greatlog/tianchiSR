import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import os.path as osp
import glob
import sys
sys.path.insert(0, '../../')
import pyflow
from src.data_utils import random_resize, random_flip, random_crop_new, random_rotate, colorjitter
from IPython import embed
import numpy as np
import pickle

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,0,1)
    return flow

class DataLoader(Dataset):
    def __init__(self, args):
        super(DataLoader, self).__init__()
        self.crop_size = args.crop_size
        self.group_dir = args.group_dir
        self.nframes = args.nframes
        
        self.imglist = pickle.load(open(osp.join(self.group_dir, 'trainlist_aug.pkl'), 'rb'))
        random.shuffle(self.imglist)
        self.length = len(self.imglist)
        print("Dataset Length: {}".format(self.length))
        
    def aug(self, img, label):
        img, label = random_crop(img, label, self.crop_size)
        img, label = random_flip(img, label)
        return img, label
        
    def __getitem__(self,index):
        index = index % self.length
        lr_img = []
        hr_img = []
        for i in range(self.nframes):
            lr_img.append(cv2.imread(self.imglist[index][0][i]))

        lr_img = np.stack(lr_img, -1)
        hr_img = cv2.imread(self.imglist[index][1][self.nframes - 1])
        
        lr_img, hr_img = random_crop_new(lr_img, hr_img, self.crop_size)
        lr_img, hr_img = random_flip(lr_img, hr_img)
        lr_target = lr_img[:,:,:,-1]
        bicubic  = cv2.resize(lr_target, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        lr_target  = lr_target.transpose(2, 0, 1) / 255
        hr_img = hr_img.transpose(2, 0, 1) / 255
        neighbor = [lr_img[:,:,:,j].transpose(2,0,1)/255 for j in range(self.nframes-1)]
        bicubic = bicubic.transpose(2, 0 ,1) / 255
        
        return lr_target, hr_img, neighbor, bicubic
    
    def __len__(self):
        return self.length