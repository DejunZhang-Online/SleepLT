# Environment:Pycharm Python3.10
# @Author:张德军
# 永无Bug
# @Time:15:47
# @File:block_wavelet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWT1D, IDWT1D, DWT1DForward # 1D离散小波变换

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        Channel attention mechanism
        """
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    

class MCFLayer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, bias=False, padding=1), 
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            )

        self.dw_conv1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch))

        self.dw_conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch))

        self.dw_conv3 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=9, stride=1, padding=4, groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch))

        self.dw_conv4 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=11, stride=1, padding=5, groups=out_ch, bias=False),
            nn.BatchNorm1d(out_ch))
        
        self.pw_conv1x1 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU())

    def forward(self, x):
        x = self.pre_conv(x)

        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv1x1(x)
        return x
    

class BlockWavelet_Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, 
                 kernel_size: int = 7, 
                 stride: int = 1, 
                 wavelet='haar', 
                 skip=True,
                 first=False):
        
        super().__init__()
        print("in_ch:", in_ch, "out_ch:", out_ch, "kernel_size:", kernel_size, "stride:", stride, "wavelet:", wavelet, "skip:", skip, "first:", first)
        if not first:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False,
                          groups=in_ch),
                nn.BatchNorm1d(out_ch),
                nn.GELU())
        else:
            self.conv1 = MCFLayer(in_ch, out_ch)
        self.conv2_L = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False, groups=out_ch),

            )

        self.conv2_H = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False, groups=out_ch),
        )

        self.dwt = DWT1D(wave=wavelet)
        self.idwt = IDWT1D(wave=wavelet)

        self.skip = skip
        if skip:
            self.conv_skip = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False, groups=in_ch)

        self.conv_squeeze = nn.Conv1d(2, 2, 7, padding=3)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = F.leaky_relu(x)

        x_L, x_H = self.dwt(x)

        x_L = self.conv2_L(x_L)
        x_H = self.conv2_H(x_H[0])

        x = self.idwt((x_L, [x_H]))
        if self.skip:
            identity = self.conv_skip(identity)
            x = x + identity
        x = F.leaky_relu(x)
        return x
    
class Epoch_Level_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.select_wavelet = 'haar' 
        self.se_block = SELayer
        self.wavelet_layers = nn.Sequential(
            BlockWavelet_Down(1, 32, kernel_size=3, stride=1, wavelet=self.select_wavelet, skip=False,
                              first=True),
            self.se_block(32),
            nn.MaxPool1d(kernel_size=5, stride=5),
            BlockWavelet_Down(32, 64, stride=1, wavelet=self.select_wavelet, skip=False, first=False),
            self.se_block(64),
            nn.MaxPool1d(kernel_size=5, stride=5),
            BlockWavelet_Down(64, 128, stride=1, wavelet=self.select_wavelet, skip=False, first=False),
            self.se_block(128),
            nn.MaxPool1d(kernel_size=5, stride=5),
            BlockWavelet_Down(128, 128, stride=1, wavelet=self.select_wavelet, skip=False, first=False),
            self.se_block(128),
            
        )

    def forward(self, x):
        x = self.wavelet_layers(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x
