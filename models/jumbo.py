import math
import torch
import torch.nn as nn
from .vision_transformer import VisionTransformer, DropPath, Mlp, Block, resample_abs_pos_embed
from einops import rearrange
from timm.layers import trunc_normal_


class JumboBlock(Block):
    def __init__(self, J, **kwargs):
        super().__init__(**kwargs)
        self.J = J
        self.jumbo_dim = int(J * kwargs['dim'])
        self.norm1 = nn.LayerNorm(kwargs['dim'])
        self.norm2 = nn.LayerNorm(kwargs['dim'])
        self.norm3 = nn.LayerNorm(self.jumbo_dim)
        self.drop_path3 = DropPath(kwargs['drop_path'])
        del self.ls1
        del self.ls2

    def forward(self, x, shared_jumbo_mlp, return_attn=False):
        if return_attn:
            return self.attn(self.norm1(x), return_attn=True)
        
        # attention over all tokens
        x = x + self.drop_path1(self.attn(self.norm1(x)))

        # jumbo-only computation
        x_jumbo = rearrange(x[:, :self.J, :], "b l d -> b (l d)")
        x_jumbo = x_jumbo + self.drop_path3(shared_jumbo_mlp(self.norm3(x_jumbo)))

        # patch-only computation
        x_patches = x[:, self.J:, :]  # (bsz, num_patches, dim)
        x_patches = x_patches + self.drop_path2(self.mlp(self.norm2(x_patches)))

        # re-combine and return
        x_jumbo = rearrange(x_jumbo, "b (l d) -> b l d", d=x_patches.shape[-1])
        x = torch.cat([x_jumbo, x_patches], dim=1)
        return x


class Jumbo(VisionTransformer):
    def __init__(self, device, J, **kwargs):
        def custom_block_fn(**block_kwargs):
            return JumboBlock(J, **block_kwargs)
        
        kwargs["block_fn"] = custom_block_fn
        super().__init__(**kwargs)
        self.device = device
        self.J = J
        self.jumbo_token = nn.Parameter(torch.zeros(1, J, self.embed_dim))
        self.head = nn.Linear(int(kwargs['embed_dim'] + J * kwargs['embed_dim']), kwargs['num_classes'])
        self.norm = nn.LayerNorm(kwargs['embed_dim'])
        self.jumbo_dim = int(J * kwargs['embed_dim'])

        self.jumbo_mlp = Mlp(
            in_features=self.jumbo_dim,
            hidden_features=int(self.jumbo_dim * 4),
            act_layer=nn.GELU,
            drop=0,
        )

        self.to(self.device)

        trunc_normal_(self.jumbo_token, std=0.02)
        self.apply(self._init_weights)
        torch.nn.init.constant_(self.head.weight, 0)
        torch.nn.init.constant_(self.head.bias, -math.log(kwargs['num_classes']))

    def set_pos_embed(self, grid_size):
        default_embed = self.pos_embed.data
        if type(grid_size) is int:
            grid_size = (grid_size, grid_size)

        new_position_embeddings = resample_abs_pos_embed(
                posemb=default_embed,
                new_size=grid_size,
                old_size=self.grid_size,
            )

        self.pos_embed = torch.nn.Parameter(
            new_position_embeddings.to(self.device)
        )

    def get_attn(self, img):
        x = self.patch_embed(img)
        x = x + self.pos_embed

        x = torch.cat([self.jumbo_token.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks[:-1]:
            x = block(x, self.jumbo_mlp)

        attn = self.blocks[-1](x, self.jumbo_mlp, return_attn=True) # (bsz, heads, N, N)
        jumbo_attn = attn[:, :, :self.J, self.J:].mean(dim=(1, 2))  # (bsz, num_patches)
        jumbo_attn = rearrange(jumbo_attn, "b (h w) -> b h w", h=self.grid_size[0], w=self.grid_size[1])
        return jumbo_attn

    def forward_features(self, img):
        x = self.patch_embed(img)
        x = x + self.pos_embed

        x = torch.cat([self.jumbo_token.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x, self.jumbo_mlp)

        x = self.norm(x)
        return x

    def forward_head(self, x):
        x_jumbo = rearrange(x[:, :self.J, :], "b l d -> b (l d)")  # (bsz,  jumbo_dim)
        x_patches = x[:, self.J:, :].mean(dim=1)  # (bsz,  dim)
        return self.head(torch.cat([x_jumbo, x_patches], dim=-1))  # (bsz, num_classes)
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x