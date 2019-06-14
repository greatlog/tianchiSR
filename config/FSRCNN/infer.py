import cv2
import glob
import sys 
import os
import os.path as osp
import  xml.dom.minidom
from torch import nn
import torch
sys.path.insert(0, '../../vmaf/wrapper')
sys.path.insert(0, '../../')
from src.utils import frames_to_video, y4m2yuv
from IPython import embed
import numpy as np
import time
import argparse
from multiprocessing import Process

result_dir = 'submit_results'
if not osp.exists(result_dir):
    os.mkdir(result_dir)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_LR_dir', type=str, default = '../../dataset/submit/youku_00200_00249_l')
#     parser.add_argument('--test_HR_dir', type=str, default = '../../dataset/test_imgs/youku_00150_00199_h_GT')
    parser.add_argument('--model_path', type = str, default = None)
    parser.add_argument('--ngpu', type = int, default = 1)
    parser.add_argument('--nframes', type = int, default = 7)
    args = parser.parse_args()
    return args    

def construct_model(model_path, device):
    from model import FSRCNN
    model = FSRCNN()
    ckpt = torch.load(model_path)
    new_ckpt  = {}
    for key in ckpt:
        if key.startswith('module'):
            new_key  = key[7:]
        else:
            new_key = key
        new_ckpt[new_key] = ckpt[key]
    model = model.to(device)
    model.load_state_dict(new_ckpt)
    model.eval()
    return model

def test_net(gpu_id, model_path, video_list, timeline, nframes):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:0')
    model = construct_model(model_path, device)

    for video_name in video_list:
        print('testing  ', video_name) 
        imglist = glob.glob(osp.join(test_LR_dir, video_name, '*.bmp'))
        result_img_dir = osp.join(result_dir, video_name)
        if not osp.exists(result_img_dir):
            os.mkdir(result_img_dir)
            
        for i in range(len(imglist)):
            img = cv2.imread(imglist[i]) / 255
            img_name = imglist[i].split('/')[-1]
            img = torch.as_tensor(img.transpose(2, 0, 1)[None]).float().cuda()
            HR_img = model(img).cpu().detach().numpy() * 255
                           
            HR_img  = np.squeeze(HR_img).transpose(1,2,0)
            HR_img = np.clip(HR_img, 0, 255)
            cv2.imwrite(osp.join(result_img_dir, img_name), HR_img)

        video_path = frames_to_video(result_img_dir)
    #     embed()
#         video_path = y4m2yuv(video_path)
#         HR_video_path = osp.join(test_HR_dir, video_name.replace('l','h_GT')+ '.yuv')
#         cmd = 'vmafossexec yuv420p {} {}  {} {} ../../vmaf/model/vmaf_v0.6.1.pkl   --log {}_{}.xml   --psnr --ssim --ms-ssim   --thread 2 --subsample 1'.format(1920, 1080, video_path, HR_video_path, osp.join(result_dir,video_name), timeline)
#         os.system(cmd)

def sample_video():
    results_y4m = glob.glob(osp.join(result_dir, 'Youku_*_l.y4m'))
    results_y4m.sort(key = lambda x: int(x.split('_')[2]))
    print(results_y4m)
    sample_start = int(0.1 * len(results_y4m))
    for i in range(sample_start, len(results_y4m),1):
        original_name = results_y4m[i].split('.')[0]
        new_name = original_name.replace('_l','_h')
        os.rename(original_name + '.y4m', '{}_Res.y4m'.format(new_name))
        print(new_name)
        cmd = "ffmpeg -i {}_Res.y4m -vf select='not(mod(n\,25))' -vsync 0  -y {}_Sub25_Res.y4m".format(new_name, new_name)
        os.system(cmd)

    
if __name__=='__main__':
#     args = parse_args()
#     test_LR_dir = args.test_LR_dir
# #     test_HR_dir = args.test_HR_dir
    
#     video_list  = []
#     for _ in os.listdir(test_LR_dir):
#         if osp.isdir(osp.join(test_LR_dir, _ )):
#             video_list.append(_)
            
#     sub_video_list = []
#     stride = len(video_list)//args.ngpu
#     for i in range(args.ngpu):
#         if i <args.ngpu - 1:
#             tmp = video_list[stride*i:stride*(i+1)]
#         else:
#             tmp = video_list[stride*i:]
#         sub_video_list.append(tmp)
                            
#     workers = []
#     timeline = int(time.time() * 1e6)
    
#     for gpu_id in range(args.ngpu):
#         workers.append(Process(target=test_net, args=(gpu_id, args.model_path, sub_video_list[gpu_id], timeline, args.nframes)))
    
#     for i, w in enumerate(workers):
#         print("Woker {} started".format(i))
#         w.start()
#     for i, w in enumerate(workers):
#         w.join()
#         print("Woker {} finished".format(i))
        
    sample_video()