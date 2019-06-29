import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from torchvision.utils import make_grid
import time
import copy
import os

def train_model(model1,model2, optimizer1, optimizer2, scheduler1, scheduler2, dataloader,summery_writer,device,args):
    
    Iter = 0
    model1.train()
    model2.train()
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        scheduler2.step()
        
        for index, (img, label) in enumerate(dataloader,0):
            Iter += 1
            img = img.float().to(device)
            label = label.float().to(device)

            optimizer2.zero_grad()
            b_size  = label.size(0)
            output = model2(label).view(-1)
            errD_real = - torch.log(output)



            gen_img = model1(img)
            output = model2(gen_img).view(-1)
            errD_fake = - torch.log(1 - output)

            D_loss = torch.mean(errD_fake + errD_real)

            D_loss.backward(retain_graph=True)
            optimizer2.step()

            optimizer1.zero_grad()
            ad_loss = torch.mean(-torch.log(output))
            c_loss = nn.L1Loss(reduction='mean')(gen_img, label)
            G_loss = c_loss + ad_loss*1e-3
            if index % 1==0:
                G_loss.backward()
                optimizer1.step()

            indicating = nn.MSELoss(reduction='mean')(gen_img, label)


            if Iter % args.display_fre == 0:
                lr = scheduler2.get_lr()[-1]
                print("D_Loss:{}, ad_Loss:{}, indicating:{},  lr:{}".format(D_loss.item(), ad_loss.item(), indicating.item(), lr))
                summery_writer.add_scalar('scaler/loss', D_loss.item(), Iter)
                summery_writer.add_scalar('scaler/lr', lr, Iter)
                summery_writer.add_image('images/LR', torchvision.utils.make_grid(img[:,args.nframes//2]), Iter)
                summery_writer.add_image('images/gen', torchvision.utils.make_grid(gen_img, nrow=2), Iter)
                summery_writer.add_image('images/HR', torchvision.utils.make_grid(label, nrow=2), Iter)

        
        # Each epoch has a training and validation phase
        torch.save(model2.state_dict(),os.path.join(args.save_dir,'model-D-{}.pkl'.format(epoch)))
        torch.save(model1.state_dict(),os.path.join(args.save_dir,'model-G-{}.pkl'.format(epoch)))



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
