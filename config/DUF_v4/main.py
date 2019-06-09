from config import *
import copy
from  model import G
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
    dataloaders = data.DataLoader(DataLoader(args), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    device = torch.device("cuda:0")
    
    print("constructing model ....")
    model = G(args.nframes)
        
    model = nn.DataParallel(model.to(device), gpuids)
    
#     for k,v in model.named_parameters():
#         if 'single_feature' in k:
#             v.requires_grad=False
    
    if args.resume:
        ckpt = torch.load(args.model_path)
        new_ckpt  = {}
        for key in ckpt:
            if not key.startswith('module'):
                new_key  = 'module.' + key
            else:
                new_key = key
            new_ckpt[new_key] = ckpt[key]
        model.load_state_dict(new_ckpt)
    print("model constructed")
    
    summary_writer = SummaryWriter(args.log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = ExponentialLR(optimizer,gamma=args.gamma)
    train_model(model,optimizer,scheduler,dataloaders,summary_writer,device,args)