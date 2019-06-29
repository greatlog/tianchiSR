from modules.EDVR_arch import EDVR
import torch
from torch import nn

class restore(nn.Module):
    def __init__(self, channel_num):
        super(restore, self).__init__()
        
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1 ,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1 ,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1 ,1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.trans = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1 ,1),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(256, 128, 3, 1 ,1),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(128, 64, 3, 1 ,1),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(64, 3, 3, 1 ,1)
        )
        
    def forward(self, x):
        down1 = self.down1(x)
        down2  = self.down2(down1)
        down3 = self.down3(down2)
        trans = self.trans(down3)
        up1 = self.up1(trans)
        up2 = self.up2(down2 + up1)
        up3 = self.up3(down1 + up2)
        out = x + up3
        return out
        
class stacked_EDVR(nn.Module):
    def __init__(self, nframes, num_stages):
        super(stacked_EDVR, self).__init__()
        self.nframes = nframes
        self.num_stages = num_stages
        self.edvr = EDVR(128, nframes, 8, 5, 40)
        
        self.restore = nn.ModuleList()
        for i in range(num_stages):
            self.restore.append(restore(3))
            
    def forward(self, x):
        out = []
        first_up = self.edvr(x)
        out.append(first_up)
        for i in range(self.num_stages):
            out.append(self.restore[i](out[-1]))
        return out
        