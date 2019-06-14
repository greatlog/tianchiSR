import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import os.path as osp
import glob
import sys
sys.path.insert(0, '../../')
from src.data_utils import random_resize, random_flip, random_crop_new, random_rotate, colorjitter
from IPython import embed

class DataLoader(Dataset):
    def __init__(self, args):
        super(DataLoader, self).__init__()
        self.crop_size = args.crop_size
        self.group_dir = args.group_dir
        
        self.group_list = glob.glob(osp.join(self.group_dir, '*l'))
        self.imglist = []
        for group in self.group_list:
            video_list = os.listdir(group)
            for v in video_list:
                imglist = os.listdir(osp.join(group, v))
                imglist.sort(key=lambda x:x[:-4])
                for img in imglist:
                    lr_path = osp.join(group, v, img)
                    hr_path = osp.join(group.replace('_l', '_h_GT'), v.replace('_l', '_h_GT'), img)
                    self.imglist.append([lr_path, hr_path])
        random.shuffle(self.imglist)
        self.length = len(self.imglist)
        
    def aug(self, img, label):
#         img, label = colorjitter(img, label)
#         img, label = random_resize(img, label)
#         img, label = random_rotate(img, label)
        img, label = random_crop_new(img, label, self.crop_size,0,0)
        img, label = random_flip(img, label)
        return img, label
        
    def __getitem__(self,index):
        index = index % self.length
        img = cv2.imread(self.imglist[index][0])
        label = cv2.imread(self.imglist[index][1])
#         img = img[:self.crop_size[0],:self.crop_size[1]]
#         label = label[:self.crop_size[0] * 4,:self.crop_size[1]*4]
        img, label = self.aug(img, label)
        img  = img.transpose(2, 0, 1) / 255
        label = label.transpose(2, 0, 1) / 255
        
        return img, label
    
    def __len__(self):
        return self.length