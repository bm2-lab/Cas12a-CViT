import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=(6, 34), patch_size=(2, 1), in_channels=1, embed_dim=256, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(1, 2), padding=0)
        self.conv_norm = nn.BatchNorm2d(128)
        self.patch_embed = nn.Conv2d(128, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        h_out = (image_size[0] - 1 - patch_size[0]) // patch_size[0] + 1
        w_out = ((image_size[1] - 2) // 2 - patch_size[1]) // patch_size[1] + 1
        num_patches = h_out * w_out
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_norm(x)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embedding[:, :x.size(1), :]
        return self.dropout(self.final_norm(x))


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads=12, qkv_bias=False, dropout=0.35, attention_dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def transpose_multi_head(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.transpose_multi_head, qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.35):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.fc1(x)))
        return self.dropout(self.fc2(x))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.35, attention_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout=dropout, attention_dropout=attention_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.35, attention_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(embed_dim, num_heads, mlp_ratio, dropout, attention_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=(6, 34),
                 patch_size=(2, 1),
                 in_channels=1,
                 num_classes=2,
                 embed_dim=503,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.17,
                 dropout=0.31,
                 attention_drop=0.27):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim, dropout)
        self.encoder = Encoder(embed_dim, depth, num_heads, mlp_ratio, dropout, attention_drop)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])
