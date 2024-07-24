""" Full assembly of the parts to form the complete network """
""" taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

from train.arch.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, pool_kernel_size=4):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=1)


#        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
#        self.up4 = Up(64, n_channels, bilinear)
        self.dec = DoubleDeConv(64, n_channels)
#        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.maxpool(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.dec(x)
#        x = self.up4(x, x1)
#        logits = self.outc(x)
        return x5, x
