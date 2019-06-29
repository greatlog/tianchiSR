from torchvision import models
import torch
from torch import nn

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = models.resnet18(pretrained=True)
        self.main.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        x = self.main(x)
        x = nn.functional.sigmoid(x)
        return x