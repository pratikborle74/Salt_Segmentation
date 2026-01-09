import torch.nn as nn
import torch

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64, 0.1)
        self.dconv_down2 = DoubleConv(64, 128, 0.1)
        self.dconv_down3 = DoubleConv(128, 256, 0.2)
        self.dconv_down4 = DoubleConv(256, 512, 0.3)
        self.maxpool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dconv_up3 = DoubleConv(512, 256, 0.2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dconv_up2 = DoubleConv(256, 128, 0.1)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dconv_up1 = DoubleConv(128, 64, 0.1)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.dconv_down1(x)
        c2 = self.dconv_down2(self.maxpool(c1))
        c3 = self.dconv_down3(self.maxpool(c2))
        c4 = self.dconv_down4(self.maxpool(c3))

        x = self.up3(c4); x = torch.cat([x, c3], dim=1); x = self.dconv_up3(x)
        x = self.up2(x); x = torch.cat([x, c2], dim=1); x = self.dconv_up2(x)
        x = self.up1(x); x = torch.cat([x, c1], dim=1); x = self.dconv_up1(x)

        return self.final(x)
