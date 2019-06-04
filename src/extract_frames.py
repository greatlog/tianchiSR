from utils import video_to_frames, y4m2yuv
import glob
import os.path as osp

y4m_list = glob.glob('../dataset/*/*.y4m')

for y4m in y4m_list:
    y4m = osp.abspath(y4m)
    print(y4m)
    y4m2yuv(y4m)