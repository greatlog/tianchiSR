from config import *
import copy
from  model import stacked_EDVR
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
    model = stacked_EDVR(args.nframes, 3)
        
    model = nn.DataParallel(model.to(device), gpuids)
    
    if args.resume:
        ckpt = torch.load(args.model_path)
        new_ckpt  = {}
        for key in ckpt:
            if not key.startswith('module'):
                new_key  = 'module.' + key
            else:
                new_key = 'module.edvr.' + key[7:]
            new_ckpt[new_key] = ckpt[key]
        model.load_state_dict(new_ckpt, strict=False)
    print("model constructed")
    
    for key, value in model.named_parameters():
        if 'edvr' in key:
            value.requires_grad = False
            
    summary_writer = SummaryWriter(args.log_dir)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    scheduler = ExponentialLR(optimizer,gamma=args.gamma)
    train_model(model,optimizer,scheduler,dataloaders,summary_writer,device,args)