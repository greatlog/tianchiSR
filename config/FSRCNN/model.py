from torch import nn

class resblock(nn.Module):
    def __init__(self,in_c, out_c):
        super(resblock, self).__init__()
        self.has_proj = False
        if in_c != out_c:
            self.has_proj = True
            self.skip = nn.Conv2d(in_c, out_c, 3, 1, 1)
            
        self.res = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        res = self.res(x)
        if self.has_proj:
            x = self.skip(x)
        return res + x

class FSRCNN(nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        
        self.extract_features = nn.Sequential(
            nn.Conv2d(3, 256, 9, 1, 4),
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
        
