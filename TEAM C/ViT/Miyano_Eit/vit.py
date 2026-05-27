"""
vit.py
Vision Transformer 모델 구현
논문: "An Image is Worth 16x16 Words" (Dosovitskiy et al., ICLR 2021)

구성:
  PatchEmbedding      → 이미지를 패치 시퀀스로 변환
  MultiHeadSelfAttention → Scaled Dot-Product Attention
  MLP                 → Feed-Forward Network (GELU)
  TransformerBlock    → Pre-LN + 잔차 연결
  ViT                 → 전체 모델 (CLS 토큰 → 분류)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Patch Embedding
#   z₀ = [x_class ; x¹ₚE ; x²ₚE ; ...] + E_pos
# ──────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """
    (B, C, H, W) → (B, N+1, D)

    Conv2d(kernel=P, stride=P) 로
    패치 분할 + 선형 투영을 단일 연산으로 처리한다.
    """
    def __init__(self, img_size, patch_size, in_channels, d_model, dropout):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size({img_size}) must be divisible by patch_size({patch_size})"

        self.n_patches = (img_size // patch_size) ** 2

        # 패치 분할 + 선형 투영
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )

        # [CLS] 토큰: 학습 가능한 벡터 (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 1D 위치 임베딩 (1, N+1, D)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)

        # (B, C, H, W) → (B, D, H/P, W/P) → (B, N, D)
        x = self.proj(x).flatten(2).transpose(1, 2)

        # [CLS] 토큰 확장 후 앞에 concat
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)           # (B, N+1, D)

        x = x + self.pos_embed
        return self.dropout(x)


# ──────────────────────────────────────────────
# Multi-Head Self-Attention
#   A = softmax(QKᵀ / √Dh),  out = AV
# ──────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model({d_model}) must be divisible by n_heads({n_heads})"

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5       # 1/√Dh

        self.qkv       = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj      = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape

        # Q, K, V 분리: (B, N, 3D) → (3, B, h, N, Dh)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                  # each (B, h, N, Dh)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, h, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 가중합 → reshape
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj_drop(self.proj(out))

        return out, attn   # attn: 시각화용


# ──────────────────────────────────────────────
# MLP (Feed-Forward Network)
#   Linear → GELU → Dropout → Linear → Dropout
# ──────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# Transformer Encoder Block (Pre-LN)
#   z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}
#   z_ℓ  = MLP(LN(z'_ℓ))    + z'_ℓ
# ──────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out               # 잔차 연결 1
        x = x + self.mlp(self.norm2(x))# 잔차 연결 2
        return x, attn_weights


# ──────────────────────────────────────────────
# ViT 전체 모델
# ──────────────────────────────────────────────
class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Args:
        img_size    : 입력 이미지 해상도 (정사각형 가정)
        patch_size  : 패치 크기 P
        in_channels : 입력 채널 수
        num_classes : 분류 클래스 수
        d_model     : 임베딩 차원 D
        depth       : Transformer 블록 수 L
        n_heads     : 어텐션 헤드 수
        mlp_ratio   : FFN 내부 차원 = d_model × mlp_ratio
        dropout     : 드롭아웃 비율
    """
    def __init__(
        self,
        img_size=32, patch_size=4, in_channels=3,
        num_classes=10, d_model=128, depth=6,
        n_heads=8, mlp_ratio=4, dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, d_model, dropout
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 분류 헤드: [CLS] 토큰 출력 → Linear
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        x = self.patch_embed(x)      # (B, N+1, D)

        attn_list = []
        for block in self.blocks:
            x, attn = block(x)
            attn_list.append(attn)

        x = self.norm(x)             # (B, N+1, D)
        cls_out = x[:, 0]            # [CLS] 토큰만 추출 (B, D)
        logits  = self.head(cls_out) # (B, num_classes)

        if return_attn:
            return logits, attn_list
        return logits


def build_vit(cfg: dict) -> ViT:
    """config dict로 ViT 인스턴스를 생성하는 헬퍼 함수."""
    return ViT(
        img_size    = cfg["img_size"],
        patch_size  = cfg["patch_size"],
        in_channels = cfg["in_channels"],
        num_classes = cfg["num_classes"],
        d_model     = cfg["d_model"],
        depth       = cfg["depth"],
        n_heads     = cfg["n_heads"],
        mlp_ratio   = cfg["mlp_ratio"],
        dropout     = cfg["dropout"],
    )
