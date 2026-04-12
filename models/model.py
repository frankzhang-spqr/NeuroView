"""
Defines the U-Net model architecture using PyTorch.
"""
import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class UNet(nn.Module):
    def __init__(self, in_channels=4, n_class=1):
        super().__init__()
        
        # Encoder
        self.e1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge
        self.b = conv_block(256, 512)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3 = conv_block(128, 64)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d4 = conv_block(64, 32)
        
        # Output
        self.outconv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.e1(x)
        p1 = self.pool1(s1)
        s2 = self.e2(p1)
        p2 = self.pool2(s2)
        s3 = self.e3(p2)
        p3 = self.pool3(s3)
        s4 = self.e4(p3)
        p4 = self.pool4(s4)
        
        # Bridge
        b = self.b(p4)
        
        # Decoder
        d1 = self.upconv1(b)
        d1 = torch.cat([d1, s4], dim=1)
        d1 = self.d1(d1)
        
        d2 = self.upconv2(d1)
        d2 = torch.cat([d2, s3], dim=1)
        d2 = self.d2(d2)
        
        d3 = self.upconv3(d2)
        d3 = torch.cat([d3, s2], dim=1)
        d3 = self.d3(d3)
        
        d4 = self.upconv4(d3)
        d4 = torch.cat([d4, s1], dim=1)
        d4 = self.d4(d4)
        
        return self.outconv(d4)

def build_model(in_channels=4, n_class=1):
    return UNet(in_channels, n_class)
