from torch import nn
from blocks import conv_layer
from blocks import Blocks
from blocks import ESA
from blocks import pixelshuffle_block
import torch
from torchsummary import summary


class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, depths=[1, 3, 1, 1], dims=[72, 72, 64, 64], upscale_factor=2):
        super().__init__()
        self.conv1 = conv_layer(in_chans, dims[3], 3)
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # Conv2d k4, s4
        # LayerNorm
        stem = conv_layer(dims[3], dims[0], kernel_size=3)
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = conv_layer(dims[i], dims[i + 1], kernel_size=3)
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        # 等差数列，初始值0，到drop path rate，总共depths个数
        # 构建每个stage中堆叠的block
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Blocks(dim=dims[i])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.esa = nn.ModuleList()
        for i in range(4):
            esa_layer = ESA(dims[i], nn.Conv2d)
            self.esa.append(esa_layer)
        self.conv2 = conv_layer(dims[3], dims[3], 3)
        self.upsample_block = pixelshuffle_block(dims[3], out_chans, upscale_factor)

    def forward(self, x):
        # x = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), dim=1)
        x = self.conv1(x)
        shortcut = x
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.esa[i](x)
        x = shortcut + x
        x = self.conv2(x)
        x = self.upsample_block(x)

        return x



