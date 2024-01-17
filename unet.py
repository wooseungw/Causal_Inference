import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseLighteningClass import BaseLighteningClass

#기본 Conv블럭에 잔차연결 추가
'''사용하지 않습니다.'''
class ConvBlock_redisual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock_redisual, self).__init__()
        
        # 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        
        # 3x3 convolution
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Adding the skip connection
        out += self.skip(identity)
        out = self.relu2(out)
        
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        
        # 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        # 3x3 convolution
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out
#인코더 블럭
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock,self).__init__()
        self.convblock1 = ConvBlock(in_channels, out_channels)  
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x = self.convblock1(x)
        p = self.maxpool(x)
        return x , p
#디코더 블럭
#디코더는 업샘플링 이후 스킵연결과 붙어서 convblock을 통과해야함
#skip보다 작은 x x먼저 업샘플링 32 -> 64 , skip과 결합 6464 
class DecoderBlock(nn.Module):
    def __init__(self, channels):
        super(DecoderBlock,self).__init__()
        self.upsample = nn.ConvTranspose2d(channels*2, channels, kernel_size=4, stride=2, padding='same')#x 업샘플링
        self.convblock = ConvBlock(channels*2, channels)
    def forward(self,x,skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.convblock(x)
        return x
    
class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(224*224*4, 10)
        self.fc2 = nn.Linear(10, 4) # source = 0, target = 1 회귀 가정
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        
        
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 224*224*4)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

###########################################


#Unet구조 middle의 xm값의 움직임에 주의
class Unet(nn.Module):
    def __init__(self,n_classes):
        super(Unet,self).__init__()
        self.encoder1 = EncoderBlock(3,64)
        self.encoder2 = EncoderBlock(64,128)
        self.encoder3 = EncoderBlock(128,256)
        self.encoder4 = EncoderBlock(256,512)
        
        self.middleconv = ConvBlock(512,1024)
        
        self.decoder4 = DecoderBlock(512)
        self.decoder3 = DecoderBlock(256)
        self.decoder2 = DecoderBlock(128)
        self.decoder1 = DecoderBlock(64)
        self.segmap = nn.Conv2d(64,n_classes, kernel_size=1)
        
        self.classifier = classifier()
                                        

    def forward(self,x):
        x1,p = self.encoder1(x)#3->64   #P:256,256 x1 :512,512
        x2,p = self.encoder2(p)#64->128 #P:128,128 x2:256,256
        x3,p = self.encoder3(p)#128->256#p:64,64 x3:128,128
        x4,p = self.encoder4(p)#256->512#p:32,32 x4:64,64
        
        xm = self.middleconv(p)#512->1024#32,32
        
        x = self.decoder4(xm,x4)#뉴런:1024->512->512 #출력tensor:64,64
        x = self.decoder3(x,x3)#뉴런:512->256->256 #출력tensor:128,128
        x = self.decoder2(x,x2)#뉴런:256->128->128 #출력tensor:256,256
        x = self.decoder1(x,x1)#뉴런:128->64->64 #출력tensor:512,512
        
        x_d = self.classifier(x)
        return x_d