import torch 
from torch import nn
import torch.nn.functional as Func
import numpy as np

class FR_52L(nn.Module):
    def __init__(self, uf=4, num_stage=4, nframes=7, nhb = 5):
        super(FR_52L, self).__init__()
        self.uf = uf
        self.num_stage = num_stage
        self.nframes = nframes
        self.nhb = nhb
        
        self.pre = nn.Sequential(
            nn.Conv3d(3, 128, (1, 5 ,5), 1, (0, 2, 2)),
            nn.BatchNorm3d(128),
            nn.PReLU(128),
            nn.Conv3d(128, 256, (3, 1, 1), 1, (1, 0, 0)),
            nn.BatchNorm3d(256),
            nn.PReLU(256)
        )
        
        self.feature = nn.ModuleList()
        self.out = nn.ModuleList()
        for i in range(self.num_stage):
            if i==0:
                in_c = 256
                mid_c = 128
                out_c = 32
            else:
                in_c = 288
                mid_c = 128
                out_c = 32
            
            feature = nn.Sequential(
                nn.Conv3d(in_c, mid_c, 3, 1, 1),
                nn.BatchNorm3d(mid_c),
                nn.PReLU(mid_c),
                nn.Conv3d(mid_c,out_c, 3, 1, 1),
                nn.BatchNorm3d(out_c),
                nn.PReLU(out_c)
            )
            
            out = nn.Sequential(
                nn.Conv3d(out_c, out_c, (self.nframes, 1, 1), 1),
                nn.Conv3d(out_c,  self.nframes* self.nhb**2 *self.uf**2, 1, 1)
                )
            self.out.append(out)
            self.feature.append(feature)
            
        self.f_softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pre(x)
        out = []
        features = [x]
        
        for i in range(self.num_stage):
            f = self.feature[i](features[-1])
            feature = torch.cat([x, f], dim=1)
            features.append(feature)
            
            o = self.out[i](f)
            ds_o = o.shape
            o = torch.reshape(o, [ds_o[0], self.nframes*self.nhb**2*self.uf**2, ds_o[2], ds_o[3], ds_o[4]]).permute(0, 2, 3, 4, 1)
            att = Func.softmax(torch.matmul(o.unsqueeze(5), o.unsqueeze(4)), -1)
            o = torch.matmul(o.unsqueeze(4), att).suqeeze(4).permute(0, 4, 1, 2, 3)
            o = Func.sotmax(o,1)
            out.append(o)
            
        return out

class G(nn.Module):
    def __init__(self,nframes):
        super(G, self).__init__()
        self.nframes = nframes
        self.num_stage = 4
        self.nhb = 7
        self.depth_to_space = nn.PixelShuffle(4)
        self.FR = FR_52L(nframes = nframes, num_stage=self.num_stage, nhb = self.nhb)
        
    def forward(self,x):
#         bic = Func.interpolate(x[:,:,self.nframes//2], scale_factor = 4, mode='bilinear')
        Fx = self.FR(x)
        out = []
        for i in range(self.num_stage):
            x_c = []
            for c in range(3):
                t = self.DynFilter3D(x[:, c:c+1], Fx[i][:,:,:,0], [1, self.nframes, self.nhb, self.nhb])
                t = self.depth_to_space(t)
                x_c.append(t)

            o = torch.cat(x_c, dim=1)
        out.append(o)
        return out
    
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
