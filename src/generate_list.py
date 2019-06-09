import glob
import os
import os.path as osp
import cv2
import numpy as np
import pickle
from IPython import embed

group_dir = '/data/tianchiSR/dataset/train_imgs/'
nframes = 7
thre = 50
trainlist = []
group_list = glob.glob(osp.join(group_dir, '*l'))

for group in group_list:
    video_list = os.listdir(group)
    for v in video_list:
        print(v)
        imglist = os.listdir(osp.join(group, v))
        imglist.sort(key=lambda x:x[:-4])
        for i in range(len(imglist) - nframes):
            lr_path = []
            lr_imgs = []
            hr_path = []
            for j in range(nframes):
                lr_path.append(osp.join(group, v, imglist[i + j]))
                lr_imgs.append(cv2.imread(lr_path[-1]))
                hr_path.append(osp.join(group.replace('_l', '_h_GT'), v.replace('_l', '_h_GT'), imglist[i + j]))
            
            lr_imgs = np.stack(lr_imgs, 0)
            diff = np.max(np.mean((lr_imgs - lr_imgs[nframes//2][None])**2, (1, 2, 3)))
            if diff > 10 and diff < 50:
                embed()
                continue
            trainlist.append([lr_path, hr_path])

print("Length of Datasets", len(trainlist))
with open('../dataset/trainlist_thre10.pkl','wb') as  f:
    pickle.dump(trainlist, f)