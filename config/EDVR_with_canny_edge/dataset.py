import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import os.path as osp
import glob
import sys
sys.path.insert(0, '../../')
from src.data_utils import random_resize, random_flip, random_crop_new, random_crop_edges, random_rotate, colorjitter
from IPython import embed
import numpy as np
import pickle

def edge(img):
    blurred = cv2.GaussianBlur(img,(3,3),0)
    gray=cv2.cvtColor(blurred,cv2.COLOR_RGB2GRAY)
    xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
    ygrad=cv2.Sobel(gray,cv2.CV_16SC1,0,1)
    edge_output=cv2.Canny(xgrad,ygrad,50,150)
    return edge_output[:,:,None]

class DataLoader(Dataset):
    def __init__(self, args):
        super(DataLoader, self).__init__()
        self.crop_size = args.crop_size
        self.group_dir = args.group_dir
        self.nframes = args.nframes
        
        self.imglist = pickle.load(open(osp.join(self.group_dir, 'trainvallist_aug.pkl'), 'rb'))
        random.shuffle(self.imglist)
        self.length = len(self.imglist)
        print("Dataset Length: {}".format(self.length))
        
    def aug(self, img, label):
        img, label = random_crop_new(img, label, self.crop_size, 0, 0)
        img, label = random_flip(img, label)
        return img, label
        
    def __getitem__(self,index):
        index = index % self.length
        lr_img = []
        hr_img = []
        for i in range(self.nframes):
            lr_img.append(cv2.imread(self.imglist[index][0][i]))
            hr_img.append(cv2.imread(self.imglist[index][1][i]))
        lr_img = np.stack(lr_img, -1)
        hr_img = np.stack(hr_img, -1)
        lr_img, hr_img = random_crop_new(lr_img, hr_img, self.crop_size)
        lr_img, hr_img = random_flip(lr_img, hr_img)
        edges = edge(lr_img[:,:,:,self.nframes//2]).transpose(2, 0 ,1) / 255
        lr_img  = lr_img.transpose(3, 2, 0, 1) / 255
        hr_img = hr_img.transpose(3, 2, 0, 1) / 255
        hr_img = hr_img[self.nframes//2]
        return lr_img, hr_img, edges
    
    def __len__(self):
        return self.length