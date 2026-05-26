"""
Vision Transformer from scratch.

Paper:
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

Paper-to-code mapping:
1. Image -> patch sequence
2. Patch flatten/projection -> token embedding
3. Add learnable CLS token
4. Add learnable positional embedding
5. Transformer encoder blocks
6. Use final CLS token for classification
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ViTConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    hidden_size: int = 192
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    initializer_range: float = 0.02

    @property
    def num_patches(self) -> int:
        assert self.image_size % self.patch_size == 0
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    @property
    def mlp_hidden_size(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)


class PatchEmbedding(nn.Module):
    """
    Paper Sec. 3.1:
    Split image into fixed-size patches and project each patch to D dimensions.
    Conv2d(kernel=P, stride=P) performs patchify + linear projection.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEmbedding(nn.Module):
    """
    Paper Eq. 1:
    z0 = [x_class; x_p1 E; x_p2 E; ...; x_pN E] + E_pos
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(x)
        batch_size = patch_tokens.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)

        tokens = tokens + self.pos_embed
        tokens = self.dropout(tokens)
        return tokens


class MultiHeadSelfAttention(nn.Module):
    """
    Paper Eq. 2:
    Multi-head self-attention over CLS token and patch tokens.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.attn_drop = nn.Dropout(config.attention_dropout)

        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj_drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, num_tokens, hidden_size = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)

        out = attention @ value
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, hidden_size)

        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attention:
            return out, attention
        return out, None


class MLP(nn.Module):
    """
    Paper Eq. 3:
    Position-wise feed-forward network with GELU.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_hidden_size)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(config.dropout)

        self.fc2 = nn.Linear(config.mlp_hidden_size, config.hidden_size)
        self.drop2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Paper Eq. 2-3:
    z_l' = MSA(LN(z_l-1)) + z_l-1
    z_l  = MLP(LN(z_l')) + z_l'
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadSelfAttention(config)

        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attn_out, attention = self.attn(self.norm1(x), return_attention)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x, attention


class ViTForImageClassification(nn.Module):
    """
    Full ViT:
    embeddings -> Transformer encoder -> final LN -> CLS classifier.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        self.embedding = ViTEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(config) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.num_classes)

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        x = self.embedding(x)

        attentions = []
        for block in self.blocks:
            x, attention = block(x, return_attention)
            if return_attention:
                attentions.append(attention)

        x = self.norm(x)

        cls_repr = x[:, 0]
        logits = self.head(cls_repr)

        if return_attention:
            return logits, attentions
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

        elif isinstance(module, ViTEmbedding):
            nn.init.trunc_normal_(
                module.cls_token,
                mean=0.0,
                std=self.config.initializer_range,
            )
            nn.init.trunc_normal_(
                module.pos_embed,
                mean=0.0,
                std=self.config.initializer_range,
            )


def build_vit_cifar10() -> ViTForImageClassification:
    config = ViTConfig(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        hidden_size=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
    )
    return ViTForImageClassification(config)


if __name__ == "__main__":
    model = build_vit_cifar10()
    dummy = torch.randn(2, 3, 32, 32)

    logits, attentions = model(dummy, return_attention=True)

    print("logits:", logits.shape)
    print("num_attention_layers:", len(attentions))
    print("attention_shape:", attentions[0].shape)

    assert logits.shape == (2, 10)
    assert attentions[0].shape == (2, 6, 65, 65)

    print("ViT forward test passed.")
