""" 
Code from DEIT3: https://github.com/facebookresearch/deit/blob/main/models_v2.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_scaled_dot = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.use_scaled_dot:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        init_values=1e-4,
        drop_path=0.1,
        mlp_dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
        )
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=mlp_dropout,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        block_fn=Block,
        drop_path_rate=0.1,
        mlp_dropout=0.0,
        init_values=1e-4,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate
        self.mlp_dropout = mlp_dropout
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.depth = depth
        self.init_values = init_values

        if type(img_size) is int:
            assert img_size % patch_size == 0, "img size be divisible by patch size"
            self.grid_size = (img_size // patch_size, img_size // patch_size)
        elif type(img_size) is tuple and len(img_size) == 2:
            assert img_size[0] % patch_size == 0, "img size be divisible by patch size"
            assert img_size[1] % patch_size == 0, "img size be divisible by patch size"
            self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        self.patch_embed = PatchEmbed(
            img_size=None, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )

        num_patches = self.grid_size[0] * self.grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    drop_path=dpr[i],
                    mlp_dropout=mlp_dropout,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        torch.nn.init.constant_(self.head.weight, 0)  # Set weights to 0
        if num_classes == 1_000:
            torch.nn.init.constant_(self.head.bias, -6.9)  # init at 1/1_000
        elif num_classes == 10_450:
            torch.nn.init.constant_(self.head.bias, -9.25)  # init at 1/10_450
        else:
            raise "num_classes should be 1_000 or 10_450"

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def turn_off_scaled_dot(self):
        for block in self.blocks:
            block.attn.use_scaled_dot = False

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x):
        return self.head(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

def resample_abs_pos_embed(
    posemb,
    new_size,
    old_size,
    interpolation="bicubic",
    antialias=True,
):
    # code from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py

    if new_size[0] == old_size[0] and new_size[1] == old_size[1]:
        return posemb

    # do the interpolation
    posemb = rearrange(posemb, "b (h w) c -> b c h w", h=old_size[0], w=old_size[1])
    posemb = F.interpolate(
        posemb, size=new_size, mode=interpolation, antialias=antialias
    )
    posemb = rearrange(posemb, "b c h w -> b (h w) c")

    return posemb