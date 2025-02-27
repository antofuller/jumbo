import torch
import torch.nn as nn
from models.vision_transformer import VisionTransformer, LayerScale, DropPath, Mlp, Block, resample_abs_pos_embed
from einops import rearrange
from timm.layers import trunc_normal_


class JumboBlock(Block):
    def __init__(self, J, jumbo_mlp_ratio, **kwargs):
        super().__init__(**kwargs)
        self.J = J
        self.jumbo_dim = int(J * kwargs['dim'])

        self.norm3 = nn.LayerNorm(self.jumbo_dim)
        self.jumbo_mlp = Mlp(
            in_features=self.jumbo_dim,
            hidden_features=int(self.jumbo_dim * jumbo_mlp_ratio),
            act_layer=nn.GELU,
            drop=kwargs['mlp_dropout'],
        )
        self.ls3 = LayerScale(self.jumbo_dim, init_values=kwargs['init_values'])
        self.drop_path3 = DropPath(kwargs['drop_path'])

    def forward(self, x, layer_idx, return_after_attn=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        if return_after_attn:
            return x

        x_cls = x[:, :self.J, :]  # (bsz, J, dim)
        x_cls = rearrange(x_cls, "b l d -> b (l d)")  # (bsz, J * dim)
        x_cls = x_cls + self.drop_path3(self.ls3(self.jumbo_mlp(self.norm3(x_cls))))
        if layer_idx == 11:  # hardcoded for 12 layer ViT (this covers all sizes up to ViT-Large)
            return x_cls

        x_patches = x[:, self.J:, :]  # (bsz, num_patches, dim)
        x_patches = x_patches + self.drop_path2(self.ls2(self.mlp(self.norm2(x_patches))))

        x_cls = rearrange(x_cls, "b (l d) -> b l d", d=x_patches.shape[-1])
        x = torch.cat([x_cls, x_patches], dim=1)
        return x


class Jumbo(VisionTransformer):
    def __init__(self, device, J, jumbo_mlp_ratio, **kwargs):
        def custom_block_fn(**block_kwargs):
            return JumboBlock(J, jumbo_mlp_ratio, **block_kwargs)
        
        kwargs["block_fn"] = custom_block_fn
        super().__init__(**kwargs)
        self.device = device
        self.J = J  # number of tokens to combine
        self.cls_token = nn.Parameter(torch.zeros(1, self.J, self.embed_dim))
        self.head = nn.Linear(int(J * kwargs['embed_dim']), kwargs['num_classes'])
        self.norm = nn.LayerNorm(int(J * kwargs['embed_dim']))
        self.to(self.device)

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        torch.nn.init.constant_(self.head.weight, 0)  # Set weights to 0

        if kwargs['num_classes'] == 1_000:
            torch.nn.init.constant_(self.head.bias, -6.9)  # init at 1/1_000
        elif kwargs['num_classes'] == 10_450:
            torch.nn.init.constant_(self.head.bias, -9.25)  # init at 1/10_450
        else:
            raise "num_classes should be 1_000 or 10_450"

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

    def forward_features(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        if mask_ratio is not None:
            batch_range = torch.arange(x.shape[0], device = x.device)[:, None]
            num_masked = int(mask_ratio * x.shape[1])
            rand_indices = torch.rand(x.shape[0], x.shape[1], device = x.device).argsort(dim = -1)
            unmasked_indices = rand_indices[:, num_masked:]
            x = x[batch_range, unmasked_indices]

        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        for i, block in enumerate(self.blocks):
            x = block(x, i)
        x = self.norm(x)
        return x

    def forward_head(self, x_cls):
        return self.head(x_cls)  # (bsz, num_classes)
    
    def forward(self, x, mask_ratio=None):
        x = self.forward_features(x, mask_ratio)
        x = self.forward_head(x)
        return x
        