import torch
import torch.nn as nn
from einops import repeat, rearrange
import torchsummary


class PatchEmbedding(nn.Module):
    # image_size [3,224,224]
    # patch_size [3,16,16]
    # embed_dim the size of word vector
    # dropout?

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()
        # the patches after divided
        n_patches = (image_size // patch_size) ** 2
        # 秒的地方是用卷积直接把通道数调到了最后需要的通道数
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # x_size [b,768,sqrt(n_patches),sqrt(n_patches)]
        # drop防过拟合
        self.dropout = nn.Dropout(dropout)
        # 自己在前插入的与后面等长的一个向量
        self.class_token = nn.Parameter(torch.randn((1, 1, embed_dim)))
        # 可学习参数位置函数
        self.positional_embedding = nn.Parameter(torch.randn((1, n_patches + 1, embed_dim)))

    def forward(self, x):
        # 在括号位置上进行重复
        # class_tokens = repeat(self.class_token, "() n d -> b n d", b=x.shape[0])
        class_tokens = repeat(self.class_token, "b w c -> (repeat b) w c", repeat=x.shape[0])
        # shape = [b,num,e]
        x = self.patch_embedding(x)  # [b,embed_dim,h,w]
        x = rearrange(x, "b e h w -> b (h w) e")
        x = torch.cat([class_tokens, x], dim=1)
        x = self.positional_embedding + x
        return x


class Attention(nn.Module):
    # numheads = 做多少次attention
    # head_dim 每一次的做完head_dim 的长度
    def __init__(self, embed_dim, num_heads, qkv_bias, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(
            in_features=embed_dim,
            out_features=self.all_head_dim * 3,
            bias=qkv_bias,
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, -1)  # list
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, emded_dim, mlp_ration, dropout):
        super().__init__()
        self.process = nn.Sequential(
            nn.Linear(emded_dim, int(emded_dim * mlp_ration)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emded_dim * mlp_ration), emded_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.process(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, qkv_bias, mlp_ratio, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, depth, num_classes, qkv_bias, mlp_ration, dropout):
        super().__init__()
        layer_list = []
        for i in range(depth):
            layer = EncoderLayer(embed_dim, depth, num_classes, qkv_bias, mlp_ration, dropout)
            layer_list.append(layer)
        self.process = nn.Sequential(*layer_list)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.process(x)
        x = self.norm(x)
        return x


class VisualTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, num_classes=1000, depth=3,
                 num_heads=8, mlp_ratio=4, qkv_bias=False, dropout=0., patch_dropout=0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, patch_dropout)
        self.encoder = Encoder(embed_dim, depth, num_heads, qkv_bias, mlp_ratio, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x


def main():
    sample = torch.randn((4, 3, 224, 224))
    patch_embed = VisualTransformer()
    out = patch_embed(sample)
    print(out.shape)
    print(torchsummary.summary(patch_embed, (3, 224, 224), 10))
    '''
    https://zhuanlan.zhihu.com/p/434498517
    '''


if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    main()
