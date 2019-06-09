import torch 
from torch import nn
import torch.nn.functional as Func
from torchvision import models
import numpy as np

depth_to_space = nn.PixelShuffle(4)

class FR_52L(nn.Module):
    def __init__(self, uf=4):
        super(FR_52L, self).__init__()
        self.uf = uf
        self.nframes = 7
        
        self.single_feature = models.vgg16(pretrained=False).features[:15]
        
        self.pre = nn.Conv3d(3, 64, (1, 3 ,3), 1, (0, 1, 1))
        
        F = 64
        G = 16
        
        self.stage1 = nn.ModuleList()
        for i in range(21):
            tmp = nn.Sequential(
                nn.BatchNorm3d(F),
                nn.ReLU(inplace=True),
                nn.Conv3d(F, F, 1, 1),
                nn.BatchNorm3d(F),
                nn.ReLU(inplace=True),
                nn.Conv3d(F, G, 3, 1, 1)
            )
            F = F + G
            self.stage1.append(tmp)
        
        self.stage2 = nn.ModuleList()
        for i in range(3):
            tmp = nn.Sequential(
                nn.BatchNorm3d(F),
                nn.ReLU(inplace=True),
                nn.Conv3d(F, F, 1, 1),
                nn.BatchNorm3d(F),
                nn.ReLU(inplace=True),
                nn.Conv3d(F, G, 3, 1, (0, 1, 1))
            )
            F = F + G
            self.stage2.append(tmp)
        
        self.final = nn.Sequential(
            nn.BatchNorm3d(F),
            nn.ReLU(inplace=True),
            nn.Conv3d(448, 256, (1, 3, 3), 1, (0, 1, 1)),
            nn.ReLU(inplace=True)
        )
        
        self.r = nn.Sequential(
            nn.Conv3d(256, 256, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 3*self.uf**2, 1, 1)
        )
        
        self.f = nn.Sequential(
            nn.Conv3d(256, 512, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512,  7 * 5**2 *self.uf**2, 1, 1) 
        )
        
    def forward(self, x):
        single_features = []
        for i in range(self.nframes):
            single_features.append(self.single_feature(x[:,:,i]))
        dis = ((torch.stack(single_features, -1) - single_features[self.nframes//2].unsqueeze(-1))**2).sum(1)
        dis  = Func.softmax(-Func.interpolate(dis.permute(0, 3, 1, 2), size = x.shape[3:]), 1)
        
        x = self.pre(x)
        
        for i in range(21):
            t = self.stage1[i](x)
            x = torch.cat([x, t], dim=1)
            
        for i in range(3):
            t = self.stage2[i](x)
            x = torch.cat([x[:,:,1:-1], t], dim=1)
        
        x = self.final(x)
        r = self.r(x)
        f = self.f(x)
        
        ds_f = f.shape
        f = torch.reshape(f, [ds_f[0], 7, 25, self.uf**2, ds_f[2], ds_f[3], ds_f[4]]) * dis.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        f = torch.reshape(f, [ds_f[0], 7*25, self.uf**2, ds_f[2], ds_f[3], ds_f[4]])
        f = Func.softmax(f, 1)
        
        return f, r

class G(nn.Module):
    def __init__(self,nframes):
        super(G, self).__init__()
        self.nframes = nframes
        
        self.depth_to_space = nn.PixelShuffle(4)
        self.FR = FR_52L()
        
    def forward(self,x):
#         bic = Func.interpolate(x[:,:,self.nframes//2], scale_factor = 4, mode='bilinear')
        Fx, Rx = self.FR(x)
    
        x_c = []
        for c in range(3):
            
            t = self.DynFilter3D(x[:, c:c+1], Fx[:,:,:,0])
            t = self.depth_to_space(t)
            x_c.append(t)
            
        x = torch.cat(x_c, dim=1).unsqueeze(2)
#         Rx = self.depth_to_space3D(Rx)
#         x = (x+Rx).squeeze(2)
        x = x.squeeze(2)
        return x
    
    def DynFilter3D(self, x, F, filter_size = [1, 7, 5, 5]):
        filter_local_expand = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[3], filter_size[0], np.prod(filter_size))).transpose(4, 3, 0, 1, 2)
        filter_local_expand = torch.as_tensor(filter_local_expand).float().cuda()
        x_local_expand = Func.conv3d(x, filter_local_expand, stride=1,
                                     padding = (0, (filter_size[2]-1)//2, (filter_size[3]-1)//2))# b, 1*5*5 h, w
        x_local_expand = x_local_expand.permute(0, 3, 4, 2, 1)
        F  = F.permute(0, 3, 4, 1, 2)
        x = torch.matmul(x_local_expand, F) # b, h, w, 1, R*R
        x = torch.squeeze(x, dim=3).permute(0, 3, 1, 2) # b, h, w, R*R

        return x
    
    def depth_to_space3D(self, x):
        x = x.transpose(1,2)
        ds_x = x.shape
        #(B, T, C, H, W,)
        x = torch.reshape(x, (ds_x[0]*ds_x[1], ds_x[2], ds_x[3], ds_x[4]))
        y = self.depth_to_space(x)

        ds_y = y.shape
        x = torch.reshape(y, (ds_x[0], ds_x[1], ds_y[1], ds_y[2], ds_y[3])).transpose(1,2)
        return x
