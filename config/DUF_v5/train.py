import torch.nn as nn
import torch
from torchvision import transforms
import torchvision
from torchvision.utils import make_grid
import time
import copy
import os

def Huber(y_true, y_pred, delta=torch.as_tensor(0.01).float().cuda()):
    delta = torch.as_tensor(delta).float().cuda()
    abs_error = torch.abs(y_pred - y_true)
    quadratic = torch.min(abs_error, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


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
            
#             loss = nn.MSELoss(reduction = 'mean')(label, gen_img)
            loss = Huber(gen_img, label)
            loss_ = torch.mean((gen_img - label)**2)
            loss.backward()
            optimizer.step()
            
            if Iter % args.display_fre == 0:
                lr = scheduler.get_lr()[-1]
                print("Loss:{}, lr:{}".format(loss_.item(), lr))
                summery_writer.add_scalar('scaler/loss', loss_.item(), Iter)
                summery_writer.add_scalar('scaler/lr', lr, Iter)
                summery_writer.add_image('images/LR', torchvision.utils.make_grid(img[:,:,args.nframes//2]), Iter)
                summery_writer.add_image('images/gen', torchvision.utils.make_grid(gen_img), Iter)
                summery_writer.add_image('images/HR', torchvision.utils.make_grid(label), Iter)
                
        
        # Each epoch has a training and validation phase
        torch.save(model.state_dict(),os.path.join(args.save_dir,'model-{}.pkl'.format(epoch)))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
