import cv2
import glob
import sys 
import os
import os.path as osp
import  xml.dom.minidom
sys.path.insert(0, '../../vmaf/wrapper')
sys.path.insert(0, '../../')
from src.utils import frames_to_video, y4m2yuv
from IPython import embed

result_dir = 'results'
if not osp.exists(result_dir):
    os.mkdir(result_dir)

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
        img  = cv2.imread(img_path)
        img_name = img_path.split('/')[-1]
        HR_img = cv2.resize(img, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(osp.join(result_img_dir, img_name), HR_img)
    
    video_path = frames_to_video(result_img_dir)
#     embed()
    video_path = y4m2yuv(video_path)
    HR_video_path = osp.join(test_HR_dir, video_name.replace('l','h_GT')+ '.yuv')
    cmd = 'vmafossexec yuv420p {} {}  {} {} ../../vmaf/model/vmaf_v0.6.1.pkl   --log {}.xml   --psnr --ssim --ms-ssim   --thread 2 --subsample 1'.format(1920, 1080, video_path, HR_video_path, osp.join(result_dir,video_name))
    os.system(cmd)
    
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