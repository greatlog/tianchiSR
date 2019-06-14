import glob
import os
import os.path as osp
import cv2
import numpy as np
import pickle
from IPython import embed

nframes = 7
thre = 50
trainlist = []

for imgs in ['test_imgs','train_imgs']:
    group_dir = '/data/tianchiSR/dataset/{}/'.format(imgs)
    group_list = glob.glob(osp.join(group_dir, '*l'))

    for group in group_list:
        video_list = os.listdir(group)
        for v in video_list:
            if osp.isfile(osp.join(group, v)):
                continue
            print(v)
            imglist = os.listdir(osp.join(group, v))
            imglist.sort(key=lambda x: int(x.split('/')[-1][:-4]))
            flag = True
            i = 0
            while flag:
                for k in range(3):
                    index = i + (nframes -1)*(k+1)
                    if index >= len(imglist):
                        flag = False
                        break
                    lr_path = []
                    lr_imgs = []
                    hr_path = []
                    for j in range(nframes):
                        index = i + j*(k+1)
                        lr_path.append(osp.join(group, v, imglist[index]))
                        lr_imgs.append(cv2.imread(lr_path[-1]))
                        hr_path.append(osp.join(group.replace('_l', '_h_GT'), v.replace('_l', '_h_GT'), imglist[index]))

                    lr_imgs = np.stack(lr_imgs, 0)
                    diff = np.max(np.mean((lr_imgs - lr_imgs[nframes//2][None])**2, (1, 2, 3)))
                    if diff >thre:
                        continue
                    trainlist.append([lr_path, hr_path])
                i = i+1

print("Length of Datasets", len(trainlist))
with open('../dataset/trainvallist_aug.pkl','wb') as  f:
    pickle.dump(trainlist, f)