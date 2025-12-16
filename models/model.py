import copy
import torch
import torch.nn as nn
from math import sqrt
from models.Transformer import TransformerDecoder
from models.resnet import resnet18
from models.resnet import resnet50
from einops import rearrange
from models.Cross import BASE_Transformer
from thop import profile


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding,
                                        groups=in_channels, bias=False)

        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.dw = DepthwiseSeparableConv(in_channels // 8, in_channels // 8, kernel_size=3, padding=1)

        self.max_pooling_128 = nn.MaxPool2d(kernel_size=16, stride=16)
        self.max_pooling_64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.max_pooling_32 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.max_pooling_16 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.vit = TransformerDecoder(q_dim=in_channels // 8, k_dim=in_channels // 8, v_dim=in_channels // 8,
                                          depth=1, heads=4, dim_head=32, mlp_dim=256, dropout=0, softmax=True)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=3, padding=1, bias=True)

    def forward(self, input_features_0, input_features_1):
        input_features_0 = self.conv(input_features_0)
        input_features_1 = self.conv(input_features_1)
        feature = abs(input_features_0 - input_features_1)

        # dw卷积
        input_features_0 = self.dw(input_features_0)
        input_features_1 = self.dw(input_features_1)
        query_features = self.dw(feature)

        b, c, h, w = query_features.shape

        if w == 128:
            kv_features_0 = self.max_pooling_128(input_features_0)
            kv_features_1 = self.max_pooling_128(input_features_1)
        elif w == 64:
            kv_features_0 = self.max_pooling_64(input_features_0)
            kv_features_1 = self.max_pooling_64(input_features_1)
        elif w == 32:
            kv_features_0 = self.max_pooling_32(input_features_0)
            kv_features_1 = self.max_pooling_32(input_features_1)
        elif w == 16:
            kv_features_0 = self.max_pooling_16(input_features_0)
            kv_features_1 = self.max_pooling_16(input_features_1)
        else:
            kv_features_0 = input_features_0
            kv_features_1 = input_features_1


        q = rearrange(query_features, 'b c h w -> b (h w) c')
        kv0 = rearrange(kv_features_0, 'b c h w -> b (h w) c')
        kv1 = rearrange(kv_features_1, 'b c h w -> b (h w) c')

        out0 = self.vit(q, kv0, kv0)
        out1 = self.vit(q, kv1, kv1)

        out0 = rearrange(out0, 'b (h w) c -> b c h w ', h=h)
        out1 = rearrange(out1, 'b (h w) c -> b c h w ', h=h)

        # 残差
        output_0 = input_features_0 + out0
        output_1 = input_features_1 + out1

        out = torch.cat((output_0, output_1), dim=1)

        return out

