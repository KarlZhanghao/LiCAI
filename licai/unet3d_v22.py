from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from typing import List

class conv_block(nn.Module):
    """
    Convolution Block 
    isnn: using the conv_block of nnUnet
    isres: residual UNet or not
    """
    def __init__(self, in_ch, out_ch, isnn=False, isbn=True, isres=False):
        super(conv_block, self).__init__()
        self.isres = isres
        self.s = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        if not isnn:
            if isbn:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True))
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                    nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                nn.BatchNorm3d(out_ch, eps=1e-5, affine=True, momentum=0.1),
                nn.LeakyReLU(inplace=True, negative_slope=1e-2),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='replicate'),
                nn.BatchNorm3d(out_ch, eps=1e-5, affine=True, momentum=0.1),
                nn.LeakyReLU(inplace=True, negative_slope=1e-2))

    def forward(self, inputs):
        if not self.isres:
            x = self.conv(inputs)
        else:
            x = self.conv(inputs)
            skip = self.s(inputs)
            x = x + skip
        return x
 
class unet_encoder(nn.Module):
    """
    encoder_block
    """
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.down = nn.MaxPool3d(kernel_size=(sz,2,2), stride=(sz,2,2))
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=False)
        
    def forward(self, inputs):
        x = self.down(inputs)
        x = self.conv(x)
        return x
  
class unet_decoder(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(sz,2,2), mode="trilinear", align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))            
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=False)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
          
class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=False)
        
        self.Down1 = unet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = unet_encoder(filters[1], filters[2])
        self.Down3 = unet_encoder(filters[2], filters[3])
        self.Down4 = unet_encoder(filters[3], filters[4])

        self.Up4 = unet_decoder(filters[4], filters[3])
        self.Up3 = unet_decoder(filters[3], filters[2])
        self.Up2 = unet_decoder(filters[2], filters[1])
        self.Up1 = unet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0])
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        
        d4 = e4        
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)

        d0 = self.Conv1(d0)
        out = self.Conv2(d0)

        return out

class unetnn_encoder(nn.Module):
    """"""
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.down = nn.MaxPool3d(kernel_size=(sz,2,2), stride=(sz,2,2))
        self.conv = conv_block(in_ch, out_ch, isnn=True, isres=False)
        
    def forward(self, inputs):
        x = self.down(inputs)
        x = self.conv(x)
        return x
 
class unetnn_decoder(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(sz,2,2), mode="trilinear", align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))        
        self.conv = conv_block(in_ch, out_ch, isnn=True, isres=False)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class U_Net_nn(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv0 = conv_block(n_channels, filters[0], isnn=True, isres=False)
        
        self.Down1 = unetnn_encoder(filters[0], filters[1], sz=1)
        self.Down2 = unetnn_encoder(filters[1], filters[2])
        self.Down3 = unetnn_encoder(filters[2], filters[3])
        self.Down4 = unetnn_encoder(filters[3], filters[4])

        self.Up4 = unetnn_decoder(filters[4], filters[3])
        self.Up3 = unetnn_decoder(filters[3], filters[2])
        self.Up2 = unetnn_decoder(filters[2], filters[1])
        self.Up1 = unetnn_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=True, isres=False)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        
        d4 = e4        
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)

        d0 = self.Conv1(d0)
        out = self.Conv2(d0)

        return out

class resunet_encoder(nn.Module):
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=True)
        self.down = nn.MaxPool3d(kernel_size=(sz,2,2), stride=(sz,2,2))
    def forward(self, inputs):
        x = self.down(inputs)
        x = self.conv(x)
        return x

class restunet_decoder(nn.Module):
    """
    ResUnet decoder
    """
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(sz,2,2), mode="trilinear", align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))   
        self.r = conv_block(in_ch, out_ch, isnn=False, isres=True)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x
    
class ResU_Net_D4(nn.Module):
    """
    ResUNet
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=True)
        
        self.Down1 = resunet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = resunet_encoder(filters[1], filters[2])
        self.Down3 = resunet_encoder(filters[2], filters[3])
        self.Down4 = resunet_encoder(filters[3], filters[4])

        self.Up4 = restunet_decoder(filters[4], filters[3])
        self.Up3 = restunet_decoder(filters[3], filters[2])
        self.Up2 = restunet_decoder(filters[2], filters[1])
        self.Up1 = restunet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=False, isres=True)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        
        d4 = e4        
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)

        d0 = self.Conv1(d0)
        out = self.Conv2(d0)

        return out
        
class ResU_Net_D5(nn.Module):
    """
    ResUNet
    """
    def __init__(self, n_channels=2, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        
        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=True)
        
        self.Down1 = resunet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = resunet_encoder(filters[1], filters[2], sz=1)
        self.Down3 = resunet_encoder(filters[2], filters[3])
        self.Down4 = resunet_encoder(filters[3], filters[4])
        self.Down5 = resunet_encoder(filters[4], filters[5])

        self.Up5 = restunet_decoder(filters[5], filters[4])
        self.Up4 = restunet_decoder(filters[4], filters[3])
        self.Up3 = restunet_decoder(filters[3], filters[2])
        self.Up2 = restunet_decoder(filters[2], filters[1], sz=1)
        self.Up1 = restunet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=False, isres=True)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        e5 = self.Down5(e4)
        
        d5 = e5        
        d4 = self.Up5(d5,e4)
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)

        d0 = self.Conv1(d0)
        out = self.Conv2(d0)

        return out

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
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

class attunet_decoder(nn.Module):
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(sz,2,2), mode="trilinear", align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))  
        self.attconv = Attention_block(out_ch, out_ch, out_ch)
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=False)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        g = self.attconv(x, skip)
        x = torch.cat([x,g], dim=1)#x = torch.cat([x, g], axis=1)
        x = self.conv(x)
        return x

      
class AttU_Net_D4(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=False)
        
        self.Down1 = unet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = unet_encoder(filters[1], filters[2])
        self.Down3 = unet_encoder(filters[2], filters[3])
        self.Down4 = unet_encoder(filters[3], filters[4])
        
        self.Up4 = attunet_decoder(filters[4], filters[3])
        self.Up3 = attunet_decoder(filters[3], filters[2])
        self.Up2 = attunet_decoder(filters[2], filters[1])
        self.Up1 = attunet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=False, isres=False)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        
        d4 = e4        
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)
        
        d0 = self.Conv1(d0)
        out = self.Conv2(d0)
        return out      
        
class LiCAI_AttU_Net(nn.Module):
    def __init__(self, n_organelle=1, n_channel=1, n_class=1):
        super().__init__()
        self.n_organelle = n_organelle
        self.n_channel = n_channel
        self.n_classe = n_class        
        self.attunet = nn.ModuleList([AttU_Net_D4(n_channels=n_channel, n_classes=n_class) for _ in range(n_organelle)])
    def forward(self, x):
        out = [l(x) for _, l in enumerate(self.attunet)]
        out = torch.cat(out, dim=1)
        return out


class AttU_Net_D5(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]

        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=False)
        
        self.Down1 = unet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = unet_encoder(filters[1], filters[2], sz=1)
        self.Down3 = unet_encoder(filters[2], filters[3])
        self.Down4 = unet_encoder(filters[3], filters[4])
        self.Down5 = unet_encoder(filters[4], filters[5])
        
        self.Up5 = attunet_decoder(filters[5], filters[4])
        self.Up4 = attunet_decoder(filters[4], filters[3])
        self.Up3 = attunet_decoder(filters[3], filters[2])
        self.Up2 = attunet_decoder(filters[2], filters[1], sz=1)
        self.Up1 = attunet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=False, isres=False)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        e5 = self.Down5(e4)
        
        d5 = e5        
        d4 = self.Up5(d5,e4)
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)
        
        d0 = self.Conv1(d0)
        out = self.Conv2(d0)
        return out      
        
class resattunet_decoder(nn.Module):
    def __init__(self, in_ch, out_ch, sz=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(sz,2,2), mode="trilinear", align_corners=True),
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True))  
        self.attconv = Attention_block(out_ch, out_ch, out_ch)
        self.conv = conv_block(in_ch, out_ch, isnn=False, isres=True)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        g = self.attconv(x, skip)
        x = torch.cat([x, g], axis=1)
        x = self.conv(x)
        return x
        
class ResAttU_Net_D4(nn.Module):

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv0 = conv_block(n_channels, filters[0], isnn=False, isres=True)
        
        self.Down1 = resunet_encoder(filters[0], filters[1], sz=1)
        self.Down2 = resunet_encoder(filters[1], filters[2])
        self.Down3 = resunet_encoder(filters[2], filters[3])
        self.Down4 = resunet_encoder(filters[3], filters[4])
        
        self.Up4 = resattunet_decoder(filters[4], filters[3])
        self.Up3 = resattunet_decoder(filters[3], filters[2])
        self.Up2 = resattunet_decoder(filters[2], filters[1])
        self.Up1 = resattunet_decoder(filters[1], filters[0], sz=1)

        self.Conv1 = conv_block(filters[0], filters[0], isnn=False, isres=True)
        self.Conv2 = nn.Conv3d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e0 = self.Conv0(x)
        e1 = self.Down1(e0)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)
        e4 = self.Down4(e3)
        
        d4 = e4        
        d3 = self.Up4(d4,e3)
        d2 = self.Up3(d3,e2)
        d1 = self.Up2(d2,e1)
        d0 = self.Up1(d1,e0)
        
        d0 = self.Conv1(d0)
        out = self.Conv2(d0)
        return out

if __name__=="__main__":
    model = LiCAI_AttU_Net(n_organelle=2,n_channel=2,n_class=1)
    print(model)

    x = torch.randn((1,2,16,128,128))
    out = model(x)
    print(out.shape)
        