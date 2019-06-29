from modules.EDVR_arch import EDVR
import torch
from torch import nn

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        
        self.extract_features = nn.Sequential(
            nn.Conv2d(1, 256, 5, 1, 2),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
        )
        
        self.shrink = nn.Conv2d(256, 32, 1, 1)
        
        self.map = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.PReLU(32)
        )
        
        self.expanding = nn.Conv2d(32, 256, 1, 1)
        
        self.upscale = nn.ConvTranspose2d(256, 3, 9, 4, 4, output_padding=3)
        
    def forward(self, x):
        res = nn.functional.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.extract_features(x)
        x = self.shrink(x)
        x = self.map(x) + x
        x = self.expanding(x)
        x = self.upscale(x)
        x = x + res
        return x
        
class stacked_EDVR(nn.Module):
    def __init__(self, nframes, num_stages):
        super(stacked_EDVR, self).__init__()
        self.nframes = nframes
        self.num_stages = num_stages
        self.edvr = EDVR(128, nframes, 8, 5, 40)
        
        self.edges = FSRCNN()
            
    def forward(self, x, edges):
        first_up = self.edvr(x)
        res = self.edges(edges)
        return first_up + res
        