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
import pickle
import pyflow

result_dir = 'results'
if not osp.exists(result_dir):
    os.mkdir(result_dir)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_LR_dir', type=str, default = '../../dataset/test_imgs/youku_00150_00199_l')
    parser.add_argument('--test_HR_dir', type=str, default = '../../dataset/test_imgs/youku_00150_00199_h_GT')
    parser.add_argument('--model_path', type = str, default = None)
    parser.add_argument('--ngpu', type = int, default = 1)
    parser.add_argument('--nframes', type = int, default = 7)
    args = parser.parse_args()
    return args   

def compute_psnr(hr, label):
    mse = np.mean((hr - label)**2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

def construct_model(model_path, device, nframes):
    from rbpn import Net as RBPN
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=nframes, scale_factor=4) 
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

def compute_hr(img, model, nframes, crop_size = [64, 64]):
    h = img.shape[0]
    w = img.shape[1]
    ch, cw = crop_size
    HR_img = np.zeros(shape=(1,3, h*4, w*4))
    
    lr_target = img[:,:,:,-1]
#    flow = [get_flow(lr_target, img[:,:,:,j]) for j in range(nframes-1)]
    bic = cv2.resize(lr_target, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    start_row = [_*ch for _ in range(h//ch)]
    start_row.append(h - ch)
    start_col = [_*cw for _ in range(w//cw)]
    start_col.append(w - cw)

    for row in start_row:
        for col in start_col:
            lr_patch = torch.as_tensor(lr_target[row:row+ch, col:col+cw].transpose(2,0,1)[None]).float().cuda() / 255
            neighbor = [torch.as_tensor(img[row:row+ch, col:col+cw,:,j].transpose(2,0,1)[None]).float().cuda() / 255 for j in range(nframes -1)]
 #           flow_patch = [torch.as_tensor(f[row:row+ch, col:col+cw].transpose(2,0,1)[None]).float().cuda() for f in flow]
            HR_img[:,:,row*4:row*4+ch*4, col*4:col*4 + cw*4] = model(lr_patch, neighbor).detach().cpu().numpy()
            
    HR_img  = (np.squeeze(HR_img).transpose(1,2,0) + bic / 255) * 255
    HR_img = np.clip(HR_img, 0, 255)
    return HR_img

def test_net(gpu_id, model_path, video_list, timeline, nframes):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    device = torch.device('cuda:0')
    model = construct_model(model_path, device, nframes)
    video_psnr = {}
    for video_name in video_list:
        print('testing  ', video_name)
        img_psnr = open(osp.join(result_dir, video_name+'.txt'),'w')
        accu_psnr = 0.0
        imglist = glob.glob(osp.join(test_LR_dir, video_name, '*.bmp'))
        imglist.sort(key=lambda x:(x.split('/')[-1][:-4]))
        result_img_dir = osp.join(result_dir, video_name)
        if not osp.exists(result_img_dir):
            os.mkdir(result_img_dir)
#         for i in range(nframes//2):
#             img = cv2.imread(imglist[i]) / 255
#             img_name = imglist[i].split('/')[-1]
#             img = cv2.resize(img, dsize=None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
#             cv2.imwrite(osp.join(result_img_dir, img_name), img*255)
            
#             img = cv2.imread(imglist[-i]) / 255
#             img_name = imglist[-i].split('/')[-1]
#             img = cv2.resize(img, dsize=None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
#             cv2.imwrite(osp.join(result_img_dir, img_name), img*255)
            
        for i in range(len(imglist) - (nframes)+1):
            img = []
            for j in range(nframes):
                index = abs(i+j) if (i+j)<100 else len(imglist)-1 - (i+j)
                img.append(cv2.imread(imglist[index]))
            img  = np.stack(img, -1)
            target_index = abs(i+nframes-1) if (i+nframes-1)<100 else len(imglist)-1 - (i+nframes-1)
            img_name = imglist[target_index].split('/')[-1]
            
            HR_img = compute_hr(img, model, nframes)
#             img_flip = img[:,::-1] - np.zeros_like(img)
#             HR_img = HR_img + compute_hr(img_flip, model)
#             img_flip = img[:,:,:,::-1] - np.zeros_like(img)
#             HR_img = HR_img + compute_hr(img_flip, model)[::-1]
#             img_flip = img[:,:,:,:,::-1] - np.zeros_like(img)
#             HR_img = HR_img + compute_hr(img_flip, model)[:,::-1]
            
#             HR_img = HR_img / 4
            label = cv2.imread(osp.join(test_HR_dir, video_name.replace('l','h_GT'), img_name))
            psnr = compute_psnr(HR_img, label)
            accu_psnr += psnr
            img_psnr.write('{}\t{}\n'.format(target_index, psnr))
            cv2.imwrite(osp.join(result_img_dir, img_name), HR_img)
        average_psnr  = accu_psnr / len(imglist)
        img_psnr.write('Aerage PSNR\t{}\n'.format(average_psnr))
        video_psnr[video_name] = average_psnr
        img_psnr.close()
        
    with open(osp.join(result_dir,'woker_{}_result_{}.pkl'.format(gpu_id, timeline)),'wb') as f:
        pickle.dump(video_psnr, f)


def compute_vmaf(timeline):
    results_xml = glob.glob(osp.join(result_dir, '*{}.xml'.format(timeline)))

    psnr = 0.0
    vmaf = 0.0
    ssim = 0.0
    ms_ssim = 0.0
    
    total_result = osp.join(result_dir,'total_result{}.txt'.format(timeline))
    result_file = open(total_result, 'w')

    for f in results_xml:
        dom = xml.dom.minidom.parse(f)
        fyi = dom.getElementsByTagName('fyi')[0]
        v = fyi.getAttribute('aggregateVMAF')
        p = fyi.getAttribute('aggregatePSNR')
        s = fyi.getAttribute('aggregateSSIM')
        ms_ = fyi.getAttribute('aggregateMS_SSIM')
        
        video_id = f.split('/')[-1].split('_')[1]
        result_file.write('{}\t{}\n'.format(video_id, p))

        vmaf += float(v)
        ssim += float(s)
        ms_ssim += float(ms_)
        psnr += float(p)

    vmaf /= len(results_xml)
    psnr /= len(results_xml)
    ssim /= len(results_xml)
    ms_ssim /= len(results_xml)

    print("Average VMAF: {} Average PSNR: {} Average SSIM: {} Average MS_SSIM: {}".format(vmaf, psnr, ssim, ms_ssim))
    result_file.write("Average VMAF: {} Average PSNR: {} Average SSIM: {} Average MS_SSIM: {}".format(vmaf, psnr, ssim, ms_ssim))
    result_file.close()
    
if __name__=='__main__':
    args = parse_args()
    test_LR_dir = args.test_LR_dir
    test_HR_dir = args.test_HR_dir
    
    video_list  = []
    for _ in os.listdir(test_LR_dir):
        if osp.isdir(osp.join(test_LR_dir, _ )):
            video_list.append(_)
            
    sub_video_list = [[] for i in range(args.ngpu)]
    stride = len(video_list)//args.ngpu
    for i in range(len(video_list)):
        sub_video_list[i%args.ngpu].append(video_list[i])
                            
    workers = []
    timeline = int(time.time() * 1e6)
    
    if args.ngpu==1:
        test_net(0, args.model_path, sub_video_list[0], timeline, args.nframes)
    else:
        for gpu_id in range(args.ngpu):
            workers.append(Process(target=test_net, args=(gpu_id, args.model_path, sub_video_list[gpu_id], timeline, args.nframes)))

        for i, w in enumerate(workers):
            print("Woker {} started".format(i))
            w.start()
        for i, w in enumerate(workers):
            w.join()
            print("Woker {} finished".format(i))
            
    summery_result = open('total_result.txt','w')
    total_result = {}
    average_psnr = 0.0
    tmp_result = glob.glob(osp.join(result_dir,'woker_*_result_{}.pkl'.format(timeline)))
    for tmp in tmp_result:
        tmp_pkl = pickle.load(open(tmp,'rb'))
        for key, value in tmp_pkl.items():
            total_result[key] = float(value)
            average_psnr += float(value)
    os.system('rm -rf *{}*'.format(timeline))
    average_psnr = average_psnr / len(total_result)
    summery_result.write('Average PSNR\t{}\n'.format(average_psnr))
    for item in sorted(total_result.items(), key=lambda x: x[1]):
        summery_result.write('{}\t{}\n'.format(item[0], item[1]))
    summery_result.close()
    print('Average PSNR\t{}'.format(average_psnr))
#     compute_vmaf(timeline)
