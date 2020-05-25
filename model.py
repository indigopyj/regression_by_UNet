import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from layer import *


## Unet 네트워크 구축하기

class UNet(nn.Module):
  def __init__(self, nch, nker, norm="bnorm", learning_type="plain"): # Unet 정의할때 필요한 레이어 생성
    super(UNet, self).__init__()

    self.learning_type = learning_type
    
    # contracting path (encoder)
    self.enc1_1 = CBR2d(in_channels=nch, out_channels=nker, norm=norm)
    self.enc1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    
    self.enc2_1 = CBR2d(in_channels=nker, out_channels=2*nker, norm=norm)
    self.enc2_2 = CBR2d(in_channels=2*nker, out_channels=2*nker, norm=norm)

    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.enc3_1 = CBR2d(in_channels=2*nker, out_channels=4*nker, norm=norm)
    self.enc3_2 = CBR2d(in_channels=4*nker, out_channels=4*nker, norm=norm)

    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.enc4_1 = CBR2d(in_channels=4*nker, out_channels=8*nker, norm=norm)
    self.enc4_2 = CBR2d(in_channels=8*nker, out_channels=8*nker, norm=norm)

    self.pool4 = nn.MaxPool2d(kernel_size=2)

    self.enc5_1 = CBR2d(in_channels=8*nker, out_channels=16*nker, norm=norm)


    # expansive path (decoder)
    self.dec5_1 = CBR2d(in_channels=16*nker, out_channels=8*nker, norm=norm)
    
    self.unpool4 = nn.ConvTranspose2d(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2)

    self.dec4_2 = CBR2d(in_channels=2*8*nker, out_channels=8*nker, norm=norm)
    self.dec4_1 = CBR2d(in_channels=8*nker, out_channels=4*nker, norm=norm)

    self.unpool3 = nn.ConvTranspose2d(in_channels=4*nker, out_channels=4*nker, kernel_size=2, stride=2)

    self.dec3_2 = CBR2d(in_channels=2*4*nker, out_channels=4*nker, norm=norm)
    self.dec3_1 = CBR2d(in_channels=4*nker, out_channels=2*nker, norm=norm)

    self.unpool2 = nn.ConvTranspose2d(in_channels=2*nker, out_channels=2*nker, kernel_size=2, stride=2)
    
    self.dec2_2 = CBR2d(in_channels=2*2*nker, out_channels=2*nker, norm=norm)
    self.dec2_1 = CBR2d(in_channels=2*nker, out_channels=nker, norm=norm)

    self.unpool1 = nn.ConvTranspose2d(in_channels=nker, out_channels=nker, kernel_size=2, stride=2)
    
    self.dec1_2 = CBR2d(in_channels=2*nker, out_channels=nker, norm=norm)
    self.dec1_1 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

    self.fc = nn.Conv2d(in_channels=nker, out_channels=nch, kernel_size=1)

    

  def forward(self, x) : # layer 연결
    enc1_1 = self.enc1_1(x)
    enc1_2 = self.enc1_2(enc1_1)
    pool1 = self.pool1(enc1_2)

    enc2_1 = self.enc2_1(pool1)
    enc2_2 = self.enc2_2(enc2_1)
    pool2 = self.pool2(enc2_2)

    enc3_1 = self.enc3_1(pool2)
    enc3_2 = self.enc3_2(enc3_1)
    pool3 = self.pool3(enc3_2)

    enc4_1 = self.enc4_1(pool3)
    enc4_2 = self.enc4_2(enc4_1)
    pool4 = self.pool4(enc4_2)

    enc5_1 = self.enc5_1(pool4)

    dec5_1 = self.dec5_1(enc5_1)

    unpool4 = self.unpool4(dec5_1)
    cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=[0:batch, 1:channel, 2:height, 3:width]
    dec4_2 = self.dec4_2(cat4)
    dec4_1 = self.dec4_1(dec4_2)

    unpool3 = self.unpool3(dec4_1)
    cat3 = torch.cat((unpool3, enc3_2), dim=1)
    dec3_2 = self.dec3_2(cat3)
    dec3_1 = self.dec3_1(dec3_2)

    unpool2 = self.unpool2(dec3_1)
    cat2 = torch.cat((unpool2, enc2_2), dim=1)
    dec2_2 = self.dec2_2(cat2)
    dec2_1 = self.dec2_1(dec2_2)

    unpool1 = self.unpool1(dec2_1)
    cat1 = torch.cat((unpool1, enc1_2), dim=1)
    dec1_2 = self.dec1_2(cat1)
    dec1_1 = self.dec1_1(dec1_2)
    
    if self.learning_type == "plain":
      x = self.fc(dec1_1)
    elif self.learning_type == "residual":
    # residual learning: net이 input과 label의 '차이'만을 학습할 수 있도록함 - regression task에서 사용.
      x = self.fc(dec1_1) + x

    return x

## Autoencoder 네트워크 구축하기

class Hourglass(nn.Module):
  def __init__(self, nch, nker, norm="bnorm", learning_type="plain"): # Unet 정의할때 필요한 레이어 생성
    super(Hourglass, self).__init__()

    self.learning_type = learning_type
    
    # contracting path (encoder)
    self.enc1_1 = CBR2d(in_channels=nch, out_channels=nker, norm=norm)
    self.enc1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    
    self.enc2_1 = CBR2d(in_channels=nker, out_channels=2*nker, norm=norm)
    self.enc2_2 = CBR2d(in_channels=2*nker, out_channels=2*nker, norm=norm)

    self.pool2 = nn.MaxPool2d(kernel_size=2)

    self.enc3_1 = CBR2d(in_channels=2*nker, out_channels=4*nker, norm=norm)
    self.enc3_2 = CBR2d(in_channels=4*nker, out_channels=4*nker, norm=norm)

    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.enc4_1 = CBR2d(in_channels=4*nker, out_channels=8*nker, norm=norm)
    self.enc4_2 = CBR2d(in_channels=8*nker, out_channels=8*nker, norm=norm)

    self.pool4 = nn.MaxPool2d(kernel_size=2)

    self.enc5_1 = CBR2d(in_channels=8*nker, out_channels=16*nker, norm=norm)


    # expansive path (decoder)
    self.dec5_1 = CBR2d(in_channels=16*nker, out_channels=8*nker, norm=norm)
    
    self.unpool4 = nn.ConvTranspose2d(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2)

    self.dec4_2 = CBR2d(in_channels=8*nker, out_channels=8*nker, norm=norm)
    self.dec4_1 = CBR2d(in_channels=8*nker, out_channels=4*nker, norm=norm)

    self.unpool3 = nn.ConvTranspose2d(in_channels=4*nker, out_channels=4*nker, kernel_size=2, stride=2)

    self.dec3_2 = CBR2d(in_channels=4*nker, out_channels=4*nker, norm=norm)
    self.dec3_1 = CBR2d(in_channels=4*nker, out_channels=2*nker, norm=norm)

    self.unpool2 = nn.ConvTranspose2d(in_channels=2*nker, out_channels=2*nker, kernel_size=2, stride=2)
    
    self.dec2_2 = CBR2d(in_channels=2*nker, out_channels=2*nker, norm=norm)
    self.dec2_1 = CBR2d(in_channels=2*nker, out_channels=nker, norm=norm)

    self.unpool1 = nn.ConvTranspose2d(in_channels=nker, out_channels=nker, kernel_size=2, stride=2)
    
    self.dec1_2 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)
    self.dec1_1 = CBR2d(in_channels=nker, out_channels=nker, norm=norm)

    self.fc = nn.Conv2d(in_channels=nker, out_channels=nch, kernel_size=1)

    

  def forward(self, x) : # layer 연결
    enc1_1 = self.enc1_1(x)
    enc1_2 = self.enc1_2(enc1_1)
    pool1 = self.pool1(enc1_2)

    enc2_1 = self.enc2_1(pool1)
    enc2_2 = self.enc2_2(enc2_1)
    pool2 = self.pool2(enc2_2)

    enc3_1 = self.enc3_1(pool2)
    enc3_2 = self.enc3_2(enc3_1)
    pool3 = self.pool3(enc3_2)

    enc4_1 = self.enc4_1(pool3)
    enc4_2 = self.enc4_2(enc4_1)
    pool4 = self.pool4(enc4_2)

    enc5_1 = self.enc5_1(pool4)

    dec5_1 = self.dec5_1(enc5_1)

    unpool4 = self.unpool4(dec5_1)
    #cat4 = torch.cat((unpool4, enc4_2), dim=1)  # dim=[0:batch, 1:channel, 2:height, 3:width]
    cat4 = unpool4
    dec4_2 = self.dec4_2(cat4)
    dec4_1 = self.dec4_1(dec4_2)

    unpool3 = self.unpool3(dec4_1)
    #cat3 = torch.cat((unpool3, enc3_2), dim=1)
    cat3 = unpool3
    dec3_2 = self.dec3_2(cat3)
    dec3_1 = self.dec3_1(dec3_2)

    unpool2 = self.unpool2(dec3_1)
    #cat2 = torch.cat((unpool2, enc2_2), dim=1)
    cat2 = unpool2
    dec2_2 = self.dec2_2(cat2)
    dec2_1 = self.dec2_1(dec2_2)

    unpool1 = self.unpool1(dec2_1)
    #cat1 = torch.cat((unpool1, enc1_2), dim=1)
    cat1 = unpool1
    dec1_2 = self.dec1_2(cat1)
    dec1_1 = self.dec1_1(dec1_2)
    
    if self.learning_type == "plain":
      x = self.fc(dec1_1)
    elif self.learning_type == "residual":
    # residual learning: net이 input과 label의 '차이'만을 학습할 수 있도록함 - regression task에서 사용.
      x = self.fc(dec1_1) + x

    return x
