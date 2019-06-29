import torch
from torch import nn

class convBnrelu(nn.Module):
    def __init__(self, in_c, out_c, kernel_size = 1, strides = 1):
        super(convBnrelu, self).__init__()
        
        self.norm = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, strides),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.norm(x)

class convBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(convBlock, self).__init__()
        
        half_channel = int(out_c/2)
        conv = [
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, half_channel ,1, 1),
            nn.BatchNorm2d(half_channel),
            nn.Conv2d(half_channel, half_channel, 3 ,1, 1),
            nn.BatchNorm2d(half_channel),
            nn.Conv2d(half_channel, out_c, 1, 1)   
        ]
        self.conv = nn.Sequential(*conv)
        

    def forward(self,x):
        return self.conv(x)
        

class residual(nn.Module):
    def __init__(self, in_c, out_c):
        super(residual, self).__init__()
        self.conv = convBlock(in_c, out_c)
        
        self.has_proj = False
        if in_c != out_c:
            self.has_proj = True
            self.skip = nn.Conv2d(in_c, out_c, 1, 1)
            
    def forward(self,x):
        res = self.conv(x)
        if self.has_proj:
            x = self.skip(x)
        return x+res

class hourglass(nn.Module):
    def __init__(self, in_c, out_c, n):
        super(hourglass, self).__init__()
        self.up1 = residual(in_c, out_c)
        self.low1 = nn.Sequential(
            nn.MaxPool2d(2),
            residual(in_c, out_c)
        )
        if n>0:
            self.low2 = hourglass(out_c, out_c, n-1)
        else:
            self.low2 = residual(out_c, out_c)
            
        self.low3 = residual(out_c, out_c)
    
    def forward(self,x):
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(low3, scale_factor=2, mode='bilinear')
        
        return up1 + up2
    
class wrapped_hourglass(nn.Module):
    def __init__(self, nFeat, outDim, nLow):
        super(wrapped_hourglass, self).__init__()
        self.drop = nn.Sequential(
            hourglass(nFeat, nFeat, nLow),
            convBnrelu(nFeat, nFeat),
            nn.Conv2d(nFeat, nFeat,1,1),
        )
        self.out = nn.Conv2d(nFeat, outDim, 1, 1)
        self.out_ = nn.Conv2d(outDim, nFeat, 1, 1)
        
    def forward(self,x):
        drop = self.drop(x)
        return nn.functional.normalize(self.out(drop)), x + drop + self.out_(self.out(drop))
            
