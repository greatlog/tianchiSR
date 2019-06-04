import argparse
import glob
import os.path as osp
import os

def parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--LR_dir', type=str)
    parser.add_argument('--HR_dir', type=str)
    parser.add_argument('--width', type=str)
    parser.add_argument('--height', type=str)
    args = parser.parse_args()
    return args

def measure(args):
    LR_list = glob.glob(osp.join(args.LR_dir, '*.yuv'))
    for LR_v in LR_list:
        video_name = LR_v.split('/')[-1].split('.')[0]
        HR_v = osp.join(args.HR_dir, video_name, '.yuv')
        cmd = 'vmafossexec yuv420p {} {}  {} {} model/vmaf_v0.6.1.pkl   --log {}.xml   --psnr --ssim --ms-ssim   --thread 0 --subsample 1'.format(args.width, args.height, LR_v, HR_v, osp.join(args.LR_dir,video_name))
        os.system('cmd')s