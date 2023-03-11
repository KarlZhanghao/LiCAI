from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


class conv_block3d(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, ksx, ksz, px, pz):
        super(conv_block3d, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(ksz,ksx,ksx), stride=1, padding=(pz,px,px), bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=(ksz,ksx,ksx), stride=1, padding=(pz,px,px), bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv3d(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, kusx, kusz, ksx, ksz, px, pz):
        super(up_conv3d, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(kusz,kusx,kusx)),
            nn.Conv3d(in_ch, out_ch, kernel_size=(ksz,ksx,ksx), stride=1, padding=(pz,px,px), bias=True),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block3D(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block3D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net3D(nn.Module):
    """
    Attention Unet 
    """
    def __init__(self, n_channels=1, n_classes=1):
        super(AttU_Net3D, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Conv1 = conv_block3d(n_channels, filters[0], 3, 3, 1, 1)        
        self.Maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.Conv2 = conv_block3d(filters[0], filters[1], 3, 3, 1, 1)
        self.Maxpool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        
        self.Conv3 = conv_block3d(filters[1], filters[2], 5, 3, 2, 1)
        self.Maxpool3 = nn.MaxPool3d(kernel_size=(2,4,4), stride=(2,4,4))
        
        self.Conv4 = conv_block3d(filters[2], filters[3], 5, 3, 2, 1)
        self.Maxpool4 = nn.MaxPool3d(kernel_size=(2,4,4), stride=(2,4,4))
        
        self.Conv5 = conv_block3d(filters[3], filters[4], 5, 3, 2, 1) 
        
        self.Up5 = up_conv3d(filters[4], filters[3], 4, 2, 5, 3, 2, 1)
        self.Att5 = Attention_block3D(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block3d(filters[4], filters[3], 5, 3, 2, 1)

        self.Up4 = up_conv3d(filters[3], filters[2], 4, 2, 5, 3, 2, 1)
        self.Att4 = Attention_block3D(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block3d(filters[3], filters[2], 5, 3, 2, 1)

        self.Up3 = up_conv3d(filters[2], filters[1], 2, 2, 3, 3, 1, 1)
        self.Att3 = Attention_block3D(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block3d(filters[2], filters[1], 3, 3, 1, 1)

        self.Up2 = up_conv3d(filters[1], filters[0], 2, 1, 3, 3, 1, 1)
        self.Att2 = Attention_block3D(F_g=filters[0], F_l=filters[0], F_int=16)
        self.Up_conv2 = conv_block3d(filters[1], filters[0], 3, 3, 1, 1)
        
        self.Conv = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
              
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5) 
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        return out