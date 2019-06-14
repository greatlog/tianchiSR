import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import os.path as osp
import glob
import sys
sys.path.insert(0, '../../')
from src.data_utils import random_resize, random_flip, random_crop, random_rotate, colorjitter
from IPython import embed
import numpy as np
import pickle

class DataLoader(Dataset):
    def __init__(self, args):
        super(DataLoader, self).__init__()
        self.crop_size = args.crop_size
        self.nframes = args.nframes
        self.group_dir = args.group_dir
        
        self.imglist = pickle.load(open(osp.join(self.group_dir, 'trainlist.pkl'), 'rb'))
#         self.group_list = glob.glob(osp.join(self.group_dir, '*l'))
#         self.imglist = []
#         for group in self.group_list:
#             video_list = os.listdir(group)
#             for v in video_list:
#                 imglist = os.listdir(osp.join(group, v))
#                 imglist.sort(key=lambda x:x[:-4])
#                 for i in range(0, len(imglist) - self.nframes):
#                     lr_path = []
#                     hr_path = []
#                     for j in range(self.nframes):
#                         lr_path.append(osp.join(group, v, imglist[i + j]))
#                         hr_path.append(osp.join(group.replace('_l', '_h_GT'), v.replace('_l', '_h_GT'), imglist[i + j]))
#                     self.imglist.append([lr_path, hr_path])
        random.shuffle(self.imglist)
        self.length = len(self.imglist)
        print("Dataset Length: {}".format(self.length))
        
    def aug(self, img, label):
#         img, label = colorjitter(img, label)
#         img, label = random_resize(img, label)
#         img, label = random_rotate(img, label)s
        img, label = random_crop(img, label, self.crop_size)
        img, label = random_flip(img, label)
        return img, label
        
    def __getitem__(self,index):
        index = index % self.length
        lr_img = []
        hr_img = []
        for i in range(self.nframes):
            lr_img.append(cv2.imread(self.imglist[index][0][i]))
            hr_img.append(cv2.imread(self.imglist[index][1][i]))
        lr_img = np.concatenate(lr_img, -1)
        hr_img = np.concatenate(hr_img, -1)
        lr_img, hr_img = random_crop(lr_img, hr_img, self.crop_size)
        lr_img  = lr_img.transpose(2, 0, 1) / 255
        hr_img = hr_img.transpose(2, 0, 1) / 255
        hr_img = hr_img[self.nframes//2*3:(self.nframes//2+1)*3]
        return lr_img, hr_img
    
    def __len__(self):
        return self.length