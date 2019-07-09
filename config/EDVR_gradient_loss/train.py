import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from torchvision.utils import make_grid
import time
import copy
import os
import torch.nn.functional as F

def gradient_1order(x, h_x=None, w_x=None):
    if h_x is None and w_x is None:
        h_x = x.size()[2]
        w_x = x.size()[3]
    r = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
    l = F.pad(x, (1, 0, 0, 0))[:, :, :, :w_x]
    t = F.pad(x, (0, 0, 1, 0))[:, :, :h_x, :]
    b = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
    xgrad = torch.pow(torch.pow((r - l) * 0.5, 2) + torch.pow((t - b) * 0.5, 2), 0.5)
    xgrad = nn.Softmax2d()(xgrad)
    return xgrad

def train_model(model, optimizer,scheduler, dataloader, summery_writer, device,args):
    
    Iter = 0
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        
        model.train()
        scheduler.step()
        
        for index, (img, label) in enumerate(dataloader,0):
            Iter += 1
            img = img.float().to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            
            gen_img = model(img)
            
#             loss = nn.L1Loss(reduction='mean')(label, gen_img)
            gradient1 = gradient_1order(label)
            gradient2 = gradient_1order(gen_img)
            
            loss = torch.mean(((label - gen_img)**2 + 1e-3)**0.5 + (gradient1 - gradient2)**2).mean()
            indicating = nn.MSELoss(reduction='mean')(label, gen_img)
            loss.backward()
            optimizer.step()
            
            if Iter % args.display_fre == 0:
                lr = scheduler.get_lr()[-1]
                print("Loss:{}, lr:{}".format(indicating.item(), lr))
                summery_writer.add_scalar('scaler/loss', loss.item(), Iter)
                summery_writer.add_scalar('scaler/lr', lr, Iter)
                summery_writer.add_image('images/LR', torchvision.utils.make_grid(img[:,args.nframes//2]), Iter)
                summery_writer.add_image('images/gen', torchvision.utils.make_grid(gen_img, nrow=2), Iter)
                summery_writer.add_image('images/HR', torchvision.utils.make_grid(label, nrow=2), Iter)
                
        
        # Each epoch has a training and validation phase
        torch.save(model.state_dict(),os.path.join(args.save_dir,'model-{}.pkl'.format(epoch)))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
