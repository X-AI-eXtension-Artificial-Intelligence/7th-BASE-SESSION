import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets

base_dir=os.path.dirname(os.path.abspath(__file__))

#기본 train param 설정하기
le=1e-3
batch_size=4
num_epoch=100

data_dir="./datasets"
ckpt_dir="./checkpoint" #train 된 네트워크가 저장이 될 폴더
log_dir='./log' #텐서보드 로그파일, 학습 기록이 저장 될 폴더

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#네트워크 구축하기

class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()
    #C:convolution B:Batch normalization R:ReLU 2d는 모름
        def CBR2d(in_channels,out_channels, kernel_size=3,stride=1,padding=1,bias=True):
            layers=[]
            layers+=[nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                               kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)]
            layers+=[nn.BatchNorm2d(num_features=out_channels)]
            layers+=[nn.ReLU()]

            cbr=nn.Sequential(*layers) # 하나의 함수로 정의하기
            
            return cbr
        #축소 과정 = encorder part 
        #enc=인코더 (좌측) , 1=첫번째 스테이지, _1=첫번째 레이어
        self.enc1_1=CBR2d(in_channels=1,out_channels=64) #첫번째 층의 파란색 화살표
        self.enc1_2=CBR2d(in_channels=64,out_channels=64)
        
        self.pool1=nn.MaxPool2d(kernel_size=2)

        self.enc2_1=CBR2d(in_channels=64,out_channels=128)
        self.enc2_2=CBR2d(in_channels=128,out_channels=128)
        
        self.pool2=nn.MaxPool2d(kernel_size=2)

        self.enc3_1=CBR2d(in_channels=128,out_channels=256) #첫번째 층의 파란색 화살표
        self.enc3_2=CBR2d(in_channels=256,out_channels=256)
        
        self.pool3=nn.MaxPool2d(kernel_size=2)

        self.enc4_1=CBR2d(in_channels=256,out_channels=512)
        self.enc4_2=CBR2d(in_channels=512,out_channels=512)
        
        self.pool4=nn.MaxPool2d(kernel_size=2)

        #최종 압축축!!
        self.enc5_1=CBR2d(in_channels=512,out_channels=1024)

        #확대 과정 = decorder part
        self.dec5_1=CBR2d(in_channels=1024,out_channels=512)
        
        self.unpool4=nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec4_2=CBR2d(in_channels=2*512,out_channels=512)
        self.dec4_1=CBR2d(in_channels=512,out_channels=256)

        self.unpool3=nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec3_2=CBR2d(in_channels=2*256,out_channels=256)
        self.dec3_1=CBR2d(in_channels=256,out_channels=128)

        self.unpool2=nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec2_2=CBR2d(in_channels=2*128,out_channels=128)
        self.dec2_1=CBR2d(in_channels=128,out_channels=64)

        self.unpool1=nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,padding=0,bias=True)

        self.dec1_2=CBR2d(in_channels=2*64,out_channels=64)
        self.dec1_1=CBR2d(in_channels=64,out_channels=64)

        self.fc=nn.Conv2d(in_channels=64,out_channels=2,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self,x):
        print(x.shape)
        enc1_1=self.enc1_1(x)
        enc1_2=self.enc1_2(enc1_1)
        pool1=self.pool1(enc1_2)

        print(pool1.shape)
        enc2_1=self.enc2_1(pool1)
        enc2_2=self.enc2_2(enc2_1)
        pool2=self.pool2(enc2_2)

        print(pool2.shape)
        enc3_1=self.enc3_1(pool2)
        enc3_2=self.enc3_2(enc3_1)
        pool3=self.pool3(enc3_2)

        print(pool3.shape)
        enc4_1=self.enc4_1(pool3)
        enc4_2=self.enc4_2(enc4_1)
        pool4=self.pool4(enc4_2)

        print(pool4.shape)
        enc5_1=self.enc5_1(pool4)
    
        dec5_1=self.dec5_1(enc5_1)
        unpool4=self.unpool4(dec5_1)

        print(unpool4.shape)
        dec4_2=self.dec4_2(torch.cat((unpool4,enc4_2)),dim=1)
        dec4_1=self.dec4_1(dec4_2)
        unpool3=self.unpool3(dec4_1)

        print(unpool3.shape)
        cat3=torch.cat((unpool3,enc3_2),dim=1)
        dec3_2=self.dec3_2(cat3)
        dec3_1=self.dec3_1(dec3_2)
        unpool2=self.unpool2(dec3_1)

        print(unpool2.shape)
        cat2=torch.cat((unpool2,enc2_2),dim=1)
        dec2_2=self.dec2_2(cat2)
        dec2_1=self.dec2_1(dec2_2)
        unpool1=self.pool1(dec2_1)

        print(unpool1.shape)
        cat1=torch.cat((unpool1,enc1_2),dim=1)
        dec1_2=self.dec1_2(cat1)
        dec1_1=self.dec1_1(dec1_2)
        
    
        x=self.fc(dec1_1)
        print(x)
        #끝 결과 
        return x

        