from config import *
import copy
from  modules.EDVR_arch import EDVR
from discriminator import D
import torch.utils.data as data
from torch import nn
from torchvision import models
from train import train_model
from tensorboardX import SummaryWriter
from dataset import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import torch
import torch.backends.cudnn as cudnn

def load_state_dict(model, path):
    ckpt = torch.load(path)
    new_ckpt  = {}
    for key in ckpt:
        if not key.startswith('module'):
            new_key  = 'module.' + key
        else:
            new_key = key
        new_ckpt[new_key] = ckpt[key]
    model.load_state_dict(new_ckpt, strict=True)
    return model
    

#training code
if args.phase=='train':
    dataloader = data.DataLoader(DataLoader(args), batch_size=args.batch_size,shuffle=True, num_workers=args.workers, pin_memory=True)
    
    device = torch.device("cuda:0")
    
    print("constructing model ....")
    model1 = EDVR(128, args.nframes, 8, 5, 40)
    model1 = nn.DataParallel(model1.to(device), gpuids)
    
    model2 = D()
    model2 = nn.DataParallel(model2.to(device), gpuids)
        
    if args.resume:
        model1 = load_state_dict(model1, args.model_path)
        model2 = load_state_dict(model2, args.dis_model_path)
    print("model constructed")
    
    optimizer1 = torch.optim.Adam(model1.parameters(), lr = args.lr)
    scheduler1 = ExponentialLR(optimizer1,gamma=args.gamma)
    
    optimizer2 = torch.optim.Adam(model2.parameters(), lr = args.lr)
    scheduler2 = ExponentialLR(optimizer2,gamma=args.gamma)

    summery_writer = SummaryWriter(args.log_dir)
    
    train_model(model1,model2, optimizer1, optimizer2, scheduler1, scheduler2, dataloader,summery_writer,device,args)