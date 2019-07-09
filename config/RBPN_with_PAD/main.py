from config import *
import copy
from rbpn import Net as RBPN
import torch.utils.data as data
from torch import nn
from train import train_model
from tensorboardX import SummaryWriter
from dataset import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

import torch
import torch.backends.cudnn as cudnn

#training code
if args.phase=='train':
    dataloaders = data.DataLoader(DataLoader(args), batch_size=args.batch_size,shuffle=True, num_workers=args.workers, pin_memory=True)
    
    device = torch.device("cuda:0")
    
    print("constructing model ....")
    model = RBPN(num_channels=3, base_filter=256,  feat = 64, num_stages=3, n_resblock=5, nFrames=args.nframes, scale_factor=4) 
        
    model = nn.DataParallel(model.to(device), gpuids)
    
    if args.resume:
        ckpt = torch.load(args.model_path)
        new_ckpt  = {}
        for key in ckpt:
            if not key.startswith('module'):
                new_key  = 'module.' + key
            else:
                new_key = key
            new_ckpt[new_key] = ckpt[key]
        model.load_state_dict(new_ckpt, strict=False)
    print("model constructed")
    
#     for key, value in model.named_parameters():
#         if not ('pre_deblur' in key):
#             value.requires_grad = False
            
    summary_writer = SummaryWriter(args.log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(optimizer,gamma=args.gamma)
    train_model(model,optimizer,scheduler,dataloaders,summary_writer,device,args)