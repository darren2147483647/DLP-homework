# Implement your ResNet34_UNet model here
# resnet程式碼參考了這篇文章的內容: https://meetonfriday.com/posts/fb19d450/
import torch
import torch.nn as nn
class residual_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super(residual_block,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),stride=stride,padding=(1,1),bias=False),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=(3,3),stride=1,padding=(1,1),bias=False),
            nn.BatchNorm2d(num_features=out_channel)
        )
        self.shortcut=shortcut
    def forward(self,x):
        shortcut=x
        x=self.conv(x)
        if self.shortcut is not None:
            shortcut=self.shortcut(shortcut)
        x=x+shortcut
        x=torch.relu(x)
        return x

class residual_part(nn.Module):
    def __init__(self,in_channel,out_channel,num_block=3,stride=1):
        super(residual_part,self).__init__()
        self.blocks=nn.ModuleList()
        self.num_block=num_block
        self.shortcut=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=(1,1),stride=stride,padding=(0,0),bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.blocks.append(residual_block(in_channel,out_channel,stride,self.shortcut))
        for i in range(self.num_block-1):
            self.blocks.append(residual_block(out_channel,out_channel))
    def forward(self,x):
        for i in range(self.num_block):
            x=self.blocks[i](x)
        return x

class resnet(nn.Module):
    def __init__(self,in_channel=3):
        super(resnet,self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))
        )
        in_channel=64
        self.hidden_channel=[64,128,256,512]
        self.hidden_num_block=[3,4,6,3]
        self.parts=nn.ModuleList()
        self.num_part=4
        for i in range(self.num_part):
            self.parts.append(residual_part(in_channel=in_channel,out_channel=self.hidden_channel[i],num_block=self.hidden_num_block[i],stride=(1 if i==0 else 2)))
            in_channel=self.hidden_channel[i]
    def forward(self,x):
        x=self.pre(x)
        skip=[]
        for i in range(self.num_part):
            x=self.parts[i](x)
            skip.append(x)
        return x, skip
'''
pre torch.Size([100, 64, 64, 64])
b torch.Size([100, 64, 64, 64])
b torch.Size([100, 64, 64, 64])
b torch.Size([100, 64, 64, 64])
p torch.Size([100, 64, 64, 64]) <
b torch.Size([100, 128, 32, 32])
b torch.Size([100, 128, 32, 32])
b torch.Size([100, 128, 32, 32])
b torch.Size([100, 128, 32, 32])
p torch.Size([100, 128, 32, 32]) <
b torch.Size([100, 256, 16, 16])
b torch.Size([100, 256, 16, 16])
b torch.Size([100, 256, 16, 16])
b torch.Size([100, 256, 16, 16])
b torch.Size([100, 256, 16, 16])
b torch.Size([100, 256, 16, 16])
p torch.Size([100, 256, 16, 16]) <
b torch.Size([100, 512, 8, 8])
b torch.Size([100, 512, 8, 8])
b torch.Size([100, 512, 8, 8])
p torch.Size([100, 512, 8, 8]) <
n torch.Size([100, 512, 8, 8])
torch.Size([100, 512, 8, 8])
'''
class twoConv_(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(twoConv_,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=True),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class convTran_(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(convTran_,self).__init__()
        self.tran=nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(2,2),stride=(2,2))
    def forward(self,x):
        return self.tran(x)
    
class unet_decoder(nn.Module):
    def __init__(self,in_channel=512,out_channel=1):
        super(unet_decoder,self).__init__()
        self.hidden_channels=[64,128,256,512]
        self.right=nn.ModuleList()
        for hidden_channel in reversed(self.hidden_channels):
            self.right.append(convTran_(in_channel=in_channel+hidden_channel,out_channel=hidden_channel))
            self.right.append(twoConv_(hidden_channel,hidden_channel))
            in_channel=hidden_channel
        self.end=nn.Sequential(
            convTran_(in_channel=in_channel,out_channel=in_channel),
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=(1,1),stride=(1,1),padding=(0,0),bias=True)
        )
    def forward(self,x,skip):#nx512x8x8
        skip.reverse()#[100, 64, 64, 64][100, 128, 32, 32][100, 256, 16, 16][100, 512, 8, 8] reverse
        for i in range(len(self.right)):
            if i%2:
                x=self.right[i](x)#nx512x16x16 nx256x32x32 nx128x64x64 nx64x128x128
            else:
                x=torch.concatenate((x,skip[i//2]),dim=1)#nx(512+512)x8x8 nx(512+256)x16x16 nx(256+128)x32x32 nx(128+64)x64x64
                x=self.right[i](x)#nx512x16x16 nx256x32x32 nx128x64x64 nx64x128x128
        x=self.end(x)#nx1x256x256
        return x

class res_unet(nn.Module):
    def __init__(self,in_channel=3,out_channel=1):
        super(res_unet,self).__init__()
        self.encoder=resnet(in_channel=in_channel)
        self.decoder=unet_decoder(in_channel=512,out_channel=out_channel)
    def forward(self,x):
        x,skip=self.encoder(x)
        x=self.decoder(x,skip)
        return x

#print("PyTorch Version:", torch.__version__)
#print("CUDA Version:", torch.version.cuda)
'''
model=res_unet(3,1)
x=torch.randn(100,3,256,256)
pred=model(x)
print(pred.shape)
'''
#assert False, "Not implemented yet!"