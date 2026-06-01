"""
Vision Transformer (ViT) - "An Image is Worth 16x16 Words" (Dosovitskiy et al., ICLR 2021)

Architecture:
  1. Patch Embedding  : 이미지를 (P×P) 크기 패치로 분할 후 선형 투영 → D차원
  2. Class Token      : BERT의 [CLS]처럼 학습 가능한 토큰을 시퀀스 앞에 추가
  3. Position Embedding: 1D 학습 가능한 위치 임베딩 추가
  4. Transformer Encoder (L회 반복):
       - LayerNorm → Multi-Head Self-Attention → Residual
       - LayerNorm → MLP (GELU, 2-layer) → Residual
  5. MLP Head         : [CLS] 토큰의 최종 표현 → 분류 출력

논문 Table 1 기본 설정:
  ViT-Base : L=12, D=768,  MLP=3072, heads=12, ~86M
  ViT-Large: L=24, D=1024, MLP=4096, heads=16, ~307M
  ViT-Huge : L=32, D=1280, MLP=5120, heads=16, ~632M
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch Embedding  (Eq. 1 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    이미지 x ∈ R^{H×W×C}  →  패치 시퀀스 ∈ R^{N×D}

    N = HW / P²  (패치 수)
    D            (잠재 벡터 차원)
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 각 패치 (P² × C) → D 차원 선형 투영
        # Conv2d(stride=P)로 구현하면 패치 분할 + 투영을 한 번에 처리
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)                # (B, D, N)
        x = x.transpose(1, 2)          # (B, N, D)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Self-Attention  (Appendix A, Eq. 5-8)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    [q, k, v] = z · U_qkv          U_qkv ∈ R^{D × 3D_h}
    A = softmax(q k^T / √D_h)      A ∈ R^{N×N}
    SA(z) = A v

    MSA(z) = [SA_1(z); SA_2(z); ...; SA_k(z)] · U_msa
    """

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V 통합 투영
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)                           # (B, N, 3D)
        qkv = qkv.reshape(B, N, 3, H, Dh)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3, B, H, N, Dh)
        q, k, v = qkv.unbind(0)                      # 각 (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        out = self.proj(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. MLP Block  (Eq. 3 in the paper)
# ─────────────────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    """
    MLP: Linear → GELU → Dropout → Linear → Dropout
    MLP 크기(mlp_dim)는 보통 4D (예: 768 → 3072 → 768)
    """

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Encoder Block  (Eq. 2, 3)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    z'_l = MSA(LN(z_{l-1})) + z_{l-1}    (Eq. 2)
    z_l  = MLP(LN(z'_l))   + z'_l        (Eq. 3)
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 attn_dropout: float = 0.0, mlp_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp   = MLPBlock(embed_dim, mlp_dim, mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # residual + MSA
        x = x + self.mlp(self.norm2(x))    # residual + MLP
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Vision Transformer  (Eq. 1-4)
# ─────────────────────────────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    ViT 전체 모델.

    Parameters
    ----------
    image_size   : 입력 이미지의 H = W (정사각형 가정)
    patch_size   : 패치 크기 P  (예: 16 → ViT-*/16, 32 → ViT-*/32)
    in_channels  : 입력 채널 수 (RGB → 3)
    num_classes  : 분류 클래스 수
    embed_dim    : 잠재 벡터 차원 D
    depth        : Transformer Encoder 반복 횟수 L
    num_heads    : MSA 헤드 수 k
    mlp_dim      : MLP 내부 차원
    attn_dropout : Attention 드롭아웃
    mlp_dropout  : MLP 드롭아웃
    emb_dropout  : 임베딩 직후 드롭아웃
    """

    def __init__(
        self,
        image_size: int   = 224,
        patch_size: int   = 16,
        in_channels: int  = 3,
        num_classes: int  = 1000,
        embed_dim: int    = 768,
        depth: int        = 12,
        num_heads: int    = 12,
        mlp_dim: int      = 3072,
        attn_dropout: float = 0.0,
        mlp_dropout: float  = 0.0,
        emb_dropout: float  = 0.0,
    ):
        super().__init__()

        # ── Patch Embedding (Eq. 1) ──────────────────────────────────────
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # ── Learnable [class] token  z^0_0 = x_class ────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ── 1D Learnable Position Embedding  E_pos ∈ R^{(N+1)×D} ────────
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.emb_dropout = nn.Dropout(emb_dropout)

        # ── Transformer Encoder (L blocks) ──────────────────────────────
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, attn_dropout, mlp_dropout)
            for _ in range(depth)
        ])

        # ── Layer Norm after encoder  (Eq. 4: y = LN(z^0_L)) ────────────
        self.norm = nn.LayerNorm(embed_dim)

        # ── MLP Head ─────────────────────────────────────────────────────
        # 파인튜닝 시 단일 선형층, 사전학습 시 은닉층 1개 MLP (여기선 파인튜닝용으로 단순 구현)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # ── Patch Embedding (Eq. 1) ──────────────────────────────────────
        x = self.patch_embed(x)                           # (B, N, D)

        # ── Prepend [class] token ────────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)            # (B, N+1, D)

        # ── Add Position Embedding ───────────────────────────────────────
        x = x + self.pos_embed                            # (B, N+1, D)
        x = self.emb_dropout(x)

        # ── Transformer Encoder ──────────────────────────────────────────
        x = self.transformer(x)                           # (B, N+1, D)

        # ── Classification Head  (Eq. 4) ─────────────────────────────────
        x = self.norm(x)
        cls_out = x[:, 0]                                 # (B, D)  — [CLS] 토큰만 사용
        logits = self.head(cls_out)                       # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6. 논문 Table 1의 공식 모델 변형 (편의 함수)
# ─────────────────────────────────────────────────────────────────────────────

def vit_base_patch16(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-B/16 — L=12, D=768, MLP=3072, heads=12 (~86M params)"""
    return VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
        num_classes=num_classes, **kwargs
    )

def vit_base_patch32(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-B/32 — L=12, D=768, MLP=3072, heads=12"""
    return VisionTransformer(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
        num_classes=num_classes, **kwargs
    )

def vit_large_patch16(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-L/16 — L=24, D=1024, MLP=4096, heads=16 (~307M params)"""
    return VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_dim=4096,
        num_classes=num_classes, **kwargs
    )

def vit_huge_patch14(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """ViT-H/14 — L=32, D=1280, MLP=5120, heads=16 (~632M params)"""
    return VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_dim=5120,
        num_classes=num_classes, **kwargs
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. 간단한 동작 확인
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = vit_base_patch16(num_classes=1000).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ViT-B/16 파라미터 수: {total_params / 1e6:.1f}M  (논문 기준: 86M)")

    # 더미 입력 (배치 2, RGB 224×224)
    dummy = torch.randn(2, 3, 224, 224, device=device)
    logits = model(dummy)
    print(f"입력 shape : {dummy.shape}")
    print(f"출력 shape : {logits.shape}  → (batch, num_classes)")

    # 각 모델 변형의 패치 수 확인
    print("\n[모델 변형별 패치 수 (224×224 기준)]")
    for name, ps in [("ViT-*/16", 16), ("ViT-*/32", 32), ("ViT-H/14", 14)]:
        n = (224 // ps) ** 2
        print(f"  {name:10s} patch_size={ps:2d}  →  N={n:4d} patches")
