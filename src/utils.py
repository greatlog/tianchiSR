import argparse
import glob
import os.path as osp
import os

def video_to_frames(video_path):
    paths = video_path.split('/')
    video_name = paths[-1].split('.')[0]
    img_path = osp.join('/', *paths[:-1], video_name)
    os.mkdir(img_path)
    cmd = 'ffmpeg -i {} -vsync 0 {}/%3d.bmp -y'.format(video_path,img_path)
    os.system(cmd)
    
def frames_to_video(img_path):
    paths = img_path.split('/')
    video_path = img_path
    cmd = 'ffmpeg -i {}/%3d.bmp  -pix_fmt yuv420p  -vsync 0 {}.y4m -y'.format(img_path, video_path)
    os.system(cmd)
    return (video_path+'.y4m')

def yuv2y4m(video_path, W, H):
    video_name = video_path.split('.')[0]
    cmd = 'ffmpeg -s {}x{} -i {}.yuv -vsync 0 {}.y4m -y'.format(W, H, video_name, video_name)
    os.system(cmd)
    
def y4m2yuv(video_path):
    video_name = video_path.split('.')[0]
    cmd = 'ffmpeg -i {}.y4m -vsync 0 {}.yuv  -y'.format(video_name, video_name)
    os.system(cmd)
    result_name = video_name + '.yuv'
    return result_name