import torch
from torch import nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = out + x
        return out


class Residual2(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, x2, x3, **kwargs):
        out = self.fn(x, x2, x3, **kwargs)
        out = out + x
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        out = self.fn(self.norm(x), **kwargs)
        return out


class PreNorm2(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(k_dim)
        self.norm3 = nn.LayerNorm(v_dim)
        self.fn = fn
    def forward(self, x, x2, x3, **kwargs):
        out = self.fn(self.norm1(x), self.norm2(x2), self.norm3(x3), **kwargs)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Cross_Attention(nn.Module):   #16x8=128
    def __init__(self, q_dim=128, k_dim=64, v_dim=64, heads=8, dim_head=16, dropout=0., softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = q_dim ** -0.5

        self.softmax = softmax
        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(k_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(v_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, q_dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v, mask=None):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        # 矩阵相乘  einsum爱因斯坦求和
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if self.softmax:
            attn = dots.softmax(dim=-1)
        else:
            attn = dots

        out = torch.einsum('bhij, bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class TransformerDecoder(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(q_dim, k_dim, v_dim, Cross_Attention(q_dim=q_dim, k_dim=k_dim, v_dim=v_dim, heads=heads,
                                                        dim_head=dim_head, dropout=dropout,
                                                        softmax=softmax))), #att
                Residual(PreNorm(q_dim, FeedForward(q_dim, mlp_dim, dropout=dropout))) #ff
            ]))

    def forward(self, q, k, v, mask=None):
        for idx, (attn, ff) in enumerate(self.layers, 1):
            x = attn(q, k, v, mask=mask)
            x = ff(x)
            if self.depth == 1 and idx == 1:  # 如果深度为1，只取第一层的输出
                return x
            elif idx == 4:  # 如果到达指定深度，返回当前层的输出
                return x


if __name__ == '__main__':
    transformer = TransformerDecoder(q_dim=64,k_dim=1280,v_dim=1280,depth=8,heads=8,
                                    dim_head=32,mlp_dim=512,dropout=0,softmax=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transformer = transformer.to(device)
    batch = 2
    q = torch.randn(batch,64,128,128)
    k = torch.randn(batch,1280,8,8)
    q = q.to(device)
    k = k.to(device)
    q = rearrange(q,'b c h w -> b (h w) c')
    k = rearrange(k, 'b c h w -> b (h w) c')
    x,y = transformer(q,k,k)
    h = 128
    x = rearrange(x, 'b (h w) c -> b c h w ', h=h)
    y = rearrange(y, 'b (h w) c -> b c h w ', h=h)
    print(x.shape)
    print(y.shape)