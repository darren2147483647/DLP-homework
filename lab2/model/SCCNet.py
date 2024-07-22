# implement SCCNet model

import torch
import torch.nn as nn

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass

class SCCNet(nn.Module):
    def __init__(self, numClasses=0, timeSample=0, Nu=0, C=0, Nc=0, Nt=0, dropoutRate=0):
        super(SCCNet, self).__init__()
        self.N=numClasses
        self.T=timeSample
        self.C=C
        self.dropoutRate=dropoutRate
        self.Nu=Nu
        self.Nc=Nc
        self.Nt=Nt
        self.conv1=nn.Conv2d(1,Nu,(C,Nt),bias=False)
        self.norm1=nn.BatchNorm2d(Nu)
        self.actf1=lambda x:x**2
        self.conv2=nn.Conv2d(Nu,Nc,(1,12),bias=False)#lazy conv permutation
        self.norm2=nn.BatchNorm2d(Nc)
        self.actf2=lambda x:x**2
        self.drop=nn.Dropout(p=dropoutRate)
        self.pool=nn.AvgPool2d((1,62),ceil_mode=True)
        self.flat=nn.Flatten()
        self.linear=nn.Linear(Nc*((timeSample-12+1)//62+1),4,bias=False)#ceil
        return
        pass

    def forward(self, x, testing=False):
        self.x=x
        #assert x.shape==(x.shape[0],1,self.C,self.T)
        self.z1=self.conv1(self.x)
        self.z2=self.norm1(self.z1)
        self.z3=self.actf1(self.z2)
        self.z4=self.conv2(self.z3)
        self.z5=self.norm2(self.z4)
        self.z6=self.actf2(self.z5)
        if not testing:
            self.z7=self.drop(self.z6)
        else:
            self.z7=self.z6
        self.z8=self.pool(self.z7)
        self.z9=self.flat(self.z8)
        self.o=self.linear(self.z9)
        self.y=torch.argmax(self.o,1)
        return self.o
        pass

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        return self.x[N,C]
        pass