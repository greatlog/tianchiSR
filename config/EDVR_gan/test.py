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

result_dir = 'results'
if not osp.exists(result_dir):
    os.mkdir(result_dir)

def construct_model():
    from model import FSRCNN
    model = FSRCNN()
    ckpt = torch.load('result/model-50.pkl')
    model = nn.DataParallel(model.cuda(), [0])
    model.load_state_dict(ckpt)
    model.eval()
    return model

def test_net(model):
    test_LR_dir = '../../dataset/test_imgs/youku_00150_00199_l'
    test_HR_dir = '../../dataset/test_imgs/youku_00150_00199_h_GT'
    video_list  = []
    for _ in os.listdir(test_LR_dir):
        if osp.isdir(osp.join(test_LR_dir, _ )):
            video_list.append(_)

    for video_name in video_list:
        print('testing  ', video_name) 
        imglist = glob.glob(osp.join(test_LR_dir, video_name, '*.bmp'))
        result_img_dir = osp.join(result_dir, video_name)
        if not osp.exists(result_img_dir):
            os.mkdir(result_img_dir)
        for img_path in imglist:
            img  = cv2.imread(img_path) / 255
            img_name = img_path.split('/')[-1]
            img = torch.as_tensor(img.transpose(2, 0 ,1)[None]).float().cuda()
            HR_img = model(img) * 255
            HR_img  = HR_img.cpu().detach().numpy().squeeze().transpose(1,2,0)
            HR_img = np.clip(HR_img, 0, 255)
            cv2.imwrite(osp.join(result_img_dir, img_name), HR_img)

        video_path = frames_to_video(result_img_dir)
    #     embed()
        video_path = y4m2yuv(video_path)
        HR_video_path = osp.join(test_HR_dir, video_name.replace('l','h_GT')+ '.yuv')
        cmd = 'vmafossexec yuv420p {} {}  {} {} ../../vmaf/model/vmaf_v0.6.1.pkl   --log {}.xml   --psnr --ssim --ms-ssim   --thread 2 --subsample 1'.format(1920, 1080, video_path, HR_video_path, osp.join(result_dir,video_name))
        os.system(cmd)

def compute_vmaf():
    results_xml = glob.glob(osp.join(result_dir, '*.xml'))

    psnr = 0.0
    vmaf = 0.0
    ssim = 0.0
    ms_ssim = 0.0

    for f in results_xml:
        dom = xml.dom.minidom.parse(f)
        fyi = dom.getElementsByTagName('fyi')[0]
        v = fyi.getAttribute('aggregateVMAF')
        p = fyi.getAttribute('aggregatePSNR')
        s = fyi.getAttribute('aggregateSSIM')
        ms_ = fyi.getAttribute('aggregateMS_SSIM')

        vmaf += float(v)
        ssim += float(s)
        ms_ssim += float(ms_)
        psnr += float(p)

    vmaf /= len(results_xml)
    psnr /= len(results_xml)
    ssim /= len(results_xml)
    ms_ssim /= len(results_xml)

    print("Average VMAF: {} Average PSNR: {} Average SSIM: {} Average MS_SSIM: {}".format(vmaf, psnr, ssim, ms_ssim))
    with open(osp.join(result_dir, 'final_result.txt'),'w') as f:
        f.write("Average VMAF: {} Average PSNR: {} Average SSIM: {} Average MS_SSIM: {}".format(vmaf, psnr, ssim, ms_ssim))
    
if __name__=='__main__':
    model = construct_model()
    test_net(model)
    compute_vmaf()