import os
import numpy as np
import cv2
import random

def random_resize(img, label):
    
    if random.random()>0.5:
        inter_options = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        height1, width1, _ = img.shape
        height2, width2, _ = label.shape
        ratio=random.uniform(0.7,1.3)
        size1  = (int(height1*ratio), int(width1*ratio))
        size2  = (int(height2*ratio), int(width2*ratio))
        
        img = cv2.resize(img, size1, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, size2, interpolation=cv2.INTER_NEAREST)
        return img, label
    else:
        return img, label

def colorjitter(img, label):
    
        saturation_multiplier = np.random.rand() + 0.5
        lightness_multiplier = np.random.rand() + 0.5
        hsv_img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.float32)
        hsv_label = np.array(cv2.cvtColor(label, cv2.COLOR_BGR2HSV), np.float32)
        
        hsv_img[:, :, 1] *= saturation_multiplier
        hsv_img[:, :, 2] *= lightness_multiplier
        hsv_label[:, :, 1] *= saturation_multiplier
        hsv_label[:, :, 2] *= lightness_multiplier
        
        hsv_img[hsv_img > 255] = 255
        hsv_img[hsv_img < 0] = 0
        hsv_img = np.array(hsv_img, np.uint8)
        
        hsv_label[hsv_label > 255] = 255
        hsv_label[hsv_label < 0] = 0
        hsv_label = np.array(hsv_label, np.uint8)
        
        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        label = cv2.cvtColor(hsv_label, cv2.COLOR_HSV2BGR)
        return img, label

def random_rotate(img, label, max_angle=40):
    height1, width1, channels1 = img.shape
    height2, width2, channels2 = label.shape
    
    angle = int(np.random.randint(-1,1)*np.random.uniform()*max_angle)
    M1 = cv2.getRotationMatrix2D((width1 / 2, height1 / 2), angle, 1)
    M2 = cv2.getRotationMatrix2D((width2 / 2, height2 / 2), angle, 1)
    
    cos = np.abs(M1[0,0])
    sin = np.abs(M1[0,1])
    n_width1 = int(height1*sin + width1*cos)
    n_height1 = int(height1*cos + width1*sin)
    M1[0,2] += n_width1/2.-width1/2.0
    M1[1,2] += n_height1/2.-height1/2.0
    
    cos = np.abs(M2[0,0])
    sin = np.abs(M2[0,1])
    n_width2 = int(height2*sin + width2*cos)
    n_height2 = int(height2*cos + width2*sin)
    M2[0,2] += n_width2/2.-width2/2.0
    M2[1,2] += n_height2/2.-height2/2.0
    
    img = cv2.warpAffine(img, M1, (n_width1, n_height1) , borderValue=(128, 128, 128))
    label = cv2.warpAffine(label, M2, (n_width2, n_height2), borderValue=(128, 128, 128))
    return img, label

def random_flip(img, label):
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
            return img, label
        else:
            return img, label

def random_crop(img, label, crop_size):
    height1, width1 = img.shape[:2]
    height2, width2 = label.shape[:2]
    
    random_x = np.random.rand()
    random_y = np.random.rand()
    
    lp_h = int((height1 - crop_size[0]) * random_x)
    lp_w = int((width1 - crop_size[1]) * random_y)
    img = img[lp_h:lp_h + crop_size[0], lp_w:lp_w + crop_size[1]]
    
    lp_h = int((height2 - crop_size[0]*4) * random_x)
    lp_w = int((width2 - crop_size[1]*4) * random_y)
    label = label[lp_h:lp_h + crop_size[0]*4, lp_w:lp_w + crop_size[1]*4]
    return img, label