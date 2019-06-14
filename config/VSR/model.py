from torch import nn
class VSR(nn.Module):
    def __init__(self, nframes):
        super(VSR, self).__init__()
        self.nframes = nframes
        self.extract_features = nn.Sequential(
            nn.Conv2d(3*7, 512, 5, 1, 2),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )
        
        self.shrink = nn.Conv2d(512, 32, 1, 1)
        
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
        
        self.expanding = nn.Conv2d(32, 512, 1, 1)
        
        self.upscale = nn.ConvTranspose2d(512, 3, 9, 4, 4, output_padding=3)
        
    def forward(self, x):
        res = nn.functional.interpolate(x[:,self.nframes//2*3:(self.nframes//2+1)*3], scale_factor=4, mode='bilinear')
        x = self.extract_features(x)
        x = self.shrink(x)
        x = self.map(x) + x
        x = self.expanding(x)
        x = self.upscale(x)
        x = x + res
        return x
        
