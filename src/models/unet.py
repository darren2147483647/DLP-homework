# Implement your UNet model here
# unet程式碼參考了這支影片的內容: https://www.youtube.com/watch?v=IHq1t7NxS8k
import torch
import torch.nn as nn
class twoConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(twoConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class maxPool(nn.Module):
    def __init__(self):
        super(maxPool,self).__init__()
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
    def forward(self,x):
        return self.pool(x)

class convTran(nn.Module):
    def __init__(self,in_channel):
        super(convTran,self).__init__()
        self.tran=nn.ConvTranspose2d(in_channels=in_channel,out_channels=in_channel//2,kernel_size=(2,2),stride=(2,2))
    def forward(self,x):
        return self.tran(x)
    
class unet(nn.Module):
    def __init__(self,in_channel=3,out_channel=1):
        super(unet,self).__init__()
        self.hidden_channels=[64,128,256,512]
        self.left=nn.ModuleList()
        for hidden_channel in self.hidden_channels:
            self.left.append(twoConv(in_channel,hidden_channel))
            in_channel=hidden_channel
        self.pool=maxPool()
        self.bottle_neck=twoConv(in_channel=in_channel,out_channel=in_channel*2)
        in_channel*=2
        self.right=nn.ModuleList()
        for hidden_channel in reversed(self.hidden_channels):
            self.right.append(convTran(in_channel=in_channel))
            self.right.append(twoConv(in_channel,hidden_channel))
            in_channel=hidden_channel
        self.end=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)
    def forward(self,x):#nx3x256x256
        save=[]
        for conv in self.left:
            x=conv(x)#nx64x256x256 nx128x128x128 nx256x64x64 nx512x32x32
            save.append(x)
            x=self.pool(x)#nx64x128x128 nx128x64x64 nx256x32x32 nx512x16x16
        save.reverse()
        x=self.bottle_neck(x)#nx1024x16x16
        for i in range(len(self.right)):
            if i%2:
                x=self.right[i](x)#nx512x32x32 nx256x64x64 nx128x128x128 nx64x256x256
            else:
                x=self.right[i](x)#nx512x32x32 nx256x64x64 nx128x128x128 nx64x256x256
                x=torch.concatenate((x,save[i//2]),dim=1)#nx(512+512)x32x32 nx(256+256)x64x64 nx(128+128)x128x128 nx(64+64)x256x256
        x=self.end(x)#nx1x256x256
        return x

#assert False, "Not implemented yet!"