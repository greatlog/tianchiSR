from Hourglass_layers import *

class HourglassModel(nn.Module):

    def __init__(self, nbStacks = 8, nFeat = 256, outDim = 16, nLow = 3, drop_rate = 0.1):
        super(HourglassModel, self).__init__()
        
        self.nbStack = nbStacks
        self.nFeat = nFeat
        self.outDim = outDim
        self.nLow = nLow
        self.drop_rate = drop_rate
        
        self.preprocessing = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            residual(64, 128),
            residual(128, self.nFeat)
        )
        
        self.hourglass = nn.ModuleList()
        for i in range(self.nbStack - 1):
            self.hourglass.append(
                wrapped_hourglass(self.nFeat, self.outDim, self.nLow, self.drop_rate)
            )
        
        self.last_stage = nn.Sequential(
            hourglass(self.nFeat, self.nFeat, self.nLow),
            convBnrelu(self.nFeat, self.nFeat),
            nn.Dropout2d(self.drop_rate),
            nn.Conv2d(self.nFeat, self.outDim,1,1)
        )
    
    def forward(self, x):
        pre = self.preprocessing(x)
        
        out = [None] * self.nbStack
        sum_ = [None] * self.nbStack
        
        out[0], sum_[0] = self.hourglass[0](pre)
        
        for i in range(1, self.nbStack - 1):
            out[i], sum_[i] = self.hourglass[i](sum_[i-1])
            
        out[self.nbStack - 1] = self.last_stage(sum_[self.nbStack - 2])
        
        return out