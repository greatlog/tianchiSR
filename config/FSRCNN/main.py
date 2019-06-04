from config import *
import copy
# from depth_CPM import CPM3D
from  model import FSRCNN
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
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    print("constructing model ....")
    model = FSRCNN()
        
    model = nn.DataParallel(model.to(device),gpuids)
    
    if args.resume:
        model.load_state_dict(torch.load(args.model_path))
    print("model constructed")
    
    summary_writer = SummaryWriter(args.log_dir)

    optimizer = torch.optim.Adam([{'params': model.module.extract_features.parameters()}, 
                                 {'params': model.module.shrink.parameters()},
                                 {'params': model.module.expanding.parameters()},
                                 {'params': model.module.upscale.parameters(), 'lr':0.1*args.lr}], lr = args.lr)
    scheduler = ExponentialLR(optimizer,gamma=args.gamma)
    best_model = train_model(model,optimizer,scheduler,dataloaders,summary_writer,device,args)
    torch.save(best_model.state_dict(),os.path.join(args.save_dir,'best-model.pkl'))
    
#infering code
if args.phase=='test':
    dataloaders = data.DataLoader(Dataset(args,"test", fetcher, transform_val), batch_size=args.batch_size,shuffle=False, num_workers=args.workers, pin_memory=True)

    print("constructing model ....")
    if args.model_name == 'hourglass':
        model = Hourglass3D(args, 1)
    elif args.model_name == 'CPM':
        model = CPM3D(1, args.joint_num, layers_per_recurrent_unit=args.uints_per_stage, num_recurrent_units=args.num_stage, output_size = args.output_size)
    print("model constructed")
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    summary_writer = SummaryWriter(log_dir=log_dir)
    
    model = nn.DataParallel(model.to(device),gpuids)
    model.load_state_dict(torch.load(args.model_path))
    inference(model,dataloaders,summary_writer,device,args)