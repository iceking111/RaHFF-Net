import torch
import torch.nn as nn
from einops import rearrange
from models.Transformer import TransformerDecoder

class BASE_Transformer(nn.Module):
    def __init__(self, in_channel1, in_channel2, token_len=36):
        super(BASE_Transformer, self).__init__()

        self.token_l = int(token_len ** 0.5)
        self.unfold = torch.nn.Unfold(kernel_size=2, padding=0, stride=1)
        self.tokens = None

        self.in_channel1 = in_channel1 // 8
        self.in_channel2 = in_channel2 // 8

        self.token_len = token_len
        self.conv_1 = nn.Conv2d(self.in_channel1 * 2, self.token_len, kernel_size=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(self.in_channel2 * 2, self.token_len, kernel_size=1, padding=0, bias=False)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel1, in_channel1 // 8, kernel_size=1, padding=0, bias=False),  # 为了降通道
            nn.BatchNorm2d(self.in_channel1),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel2, in_channel2 // 8, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channel2),
            nn.GELU(),
            nn.Dropout2d(0.1)
        )

        self.transformer_decoder1 = TransformerDecoder(q_dim=self.in_channel1 * 2,
                                                       k_dim=in_channel1 + in_channel2,
                                                       v_dim=in_channel1 + in_channel2,
                                                       depth=1, heads=1, dim_head=32, mlp_dim=256, dropout=0,
                                                       softmax=True)

        self.transformer_decoder2 = TransformerDecoder(q_dim=self.in_channel2 * 2,
                                                       k_dim=in_channel1 + in_channel2,
                                                       v_dim=in_channel1 + in_channel2,
                                                       depth=1, heads=1, dim_head=32, mlp_dim=256, dropout=0,
                                                       softmax=True)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = x
        if c == self.in_channel1 * 2:
            spatial_attention = self.conv_1(x)
        elif c == self.in_channel2 * 2:
            spatial_attention = self.conv_2(x)

        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        if c == self.in_channel1 * 2:
            x = self.transformer_decoder1(x, m, m)
        elif c == self.in_channel2 * 2:
            x = self.transformer_decoder2(x, m, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def forward(self, x1, x2, y1, y2):
        x1 = self.conv3(x1)
        x2 = self.conv4(x2)
        y1 = self.conv3(y1)
        y2 = self.conv4(y2)

        xy_sm = torch.cat((x1, y1), dim=1)
        xy_bg = torch.cat((x2, y2), dim=1)

        token1 = self._forward_semantic_tokens(xy_sm)
        token2 = self._forward_semantic_tokens(xy_bg)

        self.tokens = torch.cat((token1, token2), dim=2)

        b, l, c = self.tokens.shape
        self.tokens = self.tokens.permute(0, 2, 1).view(b, c, self.token_l, self.token_l)
        self.tokens = self.unfold(self.tokens)
        self.tokens = self.tokens.permute(0, 2, 1)

        xy_sm = self._forward_transformer_decoder(xy_sm, self.tokens)
        xy_bg = self._forward_transformer_decoder(xy_bg, self.tokens)

        return xy_sm, xy_bg