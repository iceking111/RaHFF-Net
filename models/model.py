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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = resnet18(pretrained=True)  # 调用 resnet.py 文件中的这个类
        self.upxy = nn.Upsample(size=(128, 128), mode='bilinear')

        self.sameLayer0 = Attention(64)
        self.sameLayer1 = Attention(64)
        self.sameLayer2 = Attention(128)
        self.sameLayer3 = Attention(256)
        self.sameLayer4 = Attention(512)

        self.crosslayer0 = BASE_Transformer(64, 64)
        self.crosslayer1 = BASE_Transformer(64, 128)
        self.crosslayer2 = BASE_Transformer(128, 256)
        self.crosslayer3 = BASE_Transformer(256, 512)

        self.max_pooling_layer_128 = nn.MaxPool2d(kernel_size=16, stride=16)
        self.max_pooling_layer_64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.max_pooling_layer_32 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.max_pooling_layer_16 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convxy_32 = nn.Conv2d(32, 16, kernel_size=1, padding=0)
        self.convxy_64 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.convxy_128 = nn.Conv2d(128, 16, kernel_size=1, padding=0)

        self.batchNorm2dend = nn.BatchNorm2d(1)

        self.visionTransformer = TransformerDecoder(q_dim=208, k_dim=624, v_dim=624, depth=8, heads=8, dim_head=64,
                                                    mlp_dim=256, dropout=0, softmax=True)

        self.end = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(208, 52, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(52),
            nn.ReLU(),
            nn.Conv2d(52, 2, kernel_size=3, padding=1),
            # nn.Sigmoid()
        )

    def forward(self, x1, y1):  # 输入两个imageA,imageB
        x = self.resnet(x1)  # 在这执行resnet
        y = self.resnet(y1)  # 在这执行resnet


        f = []  # 紫色
        f.append(self.sameLayer0(x[0], y[0]))
        f.append(self.sameLayer1(x[1], y[1]))
        f.append(self.sameLayer2(x[2], y[2]))
        f.append(self.sameLayer3(x[3], y[3]))
        f.append(self.sameLayer4(x[4], y[4]))

        g = []  # 绿色
        g.extend(self.crosslayer0(x[0], x[1], y[0], y[1]))
        g.extend(self.crosslayer1(x[1], x[2], y[1], y[2]))
        g.extend(self.crosslayer2(x[2], x[3], y[2], y[3]))
        g.extend(self.crosslayer3(x[3], x[4], y[3], y[4]))

        # 进行上采样，然后卷积，先获得q
        q = torch.cat((f[0], self.upxy(f[1]), self.convxy_32(self.upxy(f[2])), self.convxy_64(self.upxy(f[3])), self.convxy_128(self.upxy(f[4])),
                       g[0], self.upxy(g[1]), self.upxy(g[2]), self.convxy_32(self.upxy(g[3])), self.convxy_32(self.upxy(g[4])),
                       self.convxy_64(self.upxy(g[5])), self.convxy_64(self.upxy(g[6])), self.convxy_128(self.upxy(g[7]))), dim=1)

        # 对他们进行一个卷积操作
        f[0] = self.max_pooling_layer_128(f[0])
        f[1] = self.max_pooling_layer_64(f[1])
        f[2] = self.max_pooling_layer_32(f[2])
        f[3] = self.max_pooling_layer_16(f[3])
        g[0] = self.max_pooling_layer_128(g[0])
        g[1] = self.max_pooling_layer_64(g[1])
        g[2] = self.max_pooling_layer_64(g[2])
        g[3] = self.max_pooling_layer_32(g[3])
        g[4] = self.max_pooling_layer_32(g[4])
        g[5] = self.max_pooling_layer_16(g[5])
        g[6] = self.max_pooling_layer_16(g[6])

        kv = torch.cat((f[0], f[1], f[2], f[3], f[4], g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]), dim=1)

        q = rearrange(q, 'b c h w -> b (h w) c')
        kv = rearrange(kv, 'b c h w -> b (h w) c')

        h = 128
        y = self.visionTransformer(q, kv, kv)
        y = rearrange(y, 'b (h w) c -> b c h w ', h=h)
        y = self.end(y)
        return y



if __name__ == '__main__':
    net = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = net.to(device)
    batch = 1
    x = torch.randn(batch, 3, 256, 256)
    x = x.to(device)
    y = torch.randn(batch, 3, 256, 256)
    y = y.to(device)
    out = model(x, y)
    print(out.shape)
    print(1)

    flops, params = profile(model, inputs=(x,y))

    print(f'FLOPs: {flops / 1e9:.2f} GFLOPs')  # 将FLOPs转换为GFLOPs
    print(f'Params: {params / 1e6:.2f} M')  # 将参数量转换为百万（M）


