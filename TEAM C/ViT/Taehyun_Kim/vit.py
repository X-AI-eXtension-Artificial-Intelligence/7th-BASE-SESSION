import math
import torch
from torch import nn


# ============================================================
# GELU 활성화 함수
# ReLU 대신 쓰는 부드러운 활성화 함수. 트랜스포머 계열 모델의 표준.
# 수식: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
# ============================================================
class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# ============================================================
# 이미지 → 패치 임베딩 변환
# 이미지를 격자 형태의 패치로 쪼개서 각 패치를 벡터로 변환함.
# ex) 32x32 이미지 + patch_size=4 → 64개 패치 (8x8 격자)
# ============================================================
class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # 전체 패치 수 계산. ex) (32 // 4)^2 = 64개
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Conv2d로 패치 추출 + 벡터 변환을 한 번에 처리
        # kernel_size = stride = patch_size → 겹침 없이 패치 단위로 슬라이딩
        # 출력: (batch, hidden_size, H/patch, W/patch)
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        # (batch, channels, H, W) → (batch, hidden_size, H/p, W/p)
        x = self.projection(x)
        # (batch, hidden_size, H/p, W/p) → (batch, num_patches, hidden_size)
        # flatten(2): 공간 차원을 하나로 합침 → transpose: 시퀀스 형태로 맞춤
        x = x.flatten(2).transpose(1, 2)
        return x


# ============================================================
# 최종 임베딩 조합
# 패치 임베딩 + [CLS] 토큰 + 위치 임베딩을 합쳐서 인코더 입력을 만듦
# ============================================================
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # [CLS] 토큰: BERT에서 따온 개념. 시퀀스 전체를 대표하는 학습 가능한 벡터.
        # 최종적으로 이 토큰의 출력값으로 분류 수행함.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # 위치 임베딩: 각 패치의 위치 정보를 학습으로 부여함
        # 길이 = num_patches + 1 ([CLS] 토큰 자리 포함)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"])
        )
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # [CLS] 토큰을 배치 크기만큼 복제: (1, 1, hidden) → (batch, 1, hidden)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # [CLS] 토큰을 시퀀스 앞에 붙임: (batch, num_patches+1, hidden)
        x = torch.cat((cls_tokens, x), dim=1)

        # 위치 임베딩 더함 (브로드캐스팅으로 배치 전체에 적용)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


# ============================================================
# 단일 어텐션 헤드
# Q, K, V 프로젝션 후 스케일드 닷-프로덕트 어텐션 수행
# ============================================================
class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # 각 헤드가 독립적인 Q, K, V 프로젝션 행렬을 가짐
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key   = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 같은 입력 x로 Q, K, V를 각각 만듦 → 셀프 어텐션
        query = self.query(x)  # (batch, seq_len, head_size)
        key   = self.key(x)
        value = self.value(x)

        # 어텐션 스코어: Q * K^T / sqrt(head_size)
        # sqrt로 나누는 이유: 차원이 커질수록 내적값이 커져 softmax가 포화되는 걸 방지
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # softmax로 확률 분포로 변환 → 각 토큰이 다른 토큰에 얼마나 집중할지 결정
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 어텐션 가중치로 Value를 가중합
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


# ============================================================
# 멀티헤드 어텐션 (기본 버전)
# AttentionHead를 num_heads개 병렬로 실행 후 concat
# 헤드마다 다른 관점(패턴)을 학습할 수 있음
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # 각 헤드의 크기 = hidden_size / num_heads. ex) 48 / 4 = 12
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # 헤드 수만큼 AttentionHead 생성
        self.heads = nn.ModuleList([
            AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            for _ in range(self.num_attention_heads)
        ])

        # 모든 헤드의 출력을 concat한 뒤 다시 hidden_size로 프로젝션
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 헤드 병렬 실행
        attention_outputs = [head(x) for head in self.heads]

        # 헤드 출력들을 마지막 차원(feature)으로 concat
        attention_output = torch.cat(
            [out for out, _ in attention_outputs], dim=-1
        )

        # concat 결과를 다시 hidden_size로 압축
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            # 시각화용: 모든 헤드의 어텐션 확률을 스택해서 반환
            attention_probs = torch.stack(
                [probs for _, probs in attention_outputs], dim=1
            )
            return (attention_output, attention_probs)


# ============================================================
# 멀티헤드 어텐션 (최적화 버전)
# Q, K, V를 헤드별로 따로 계산하지 않고 한 번에 프로젝션 후 분리
# 행렬 연산을 배치 처리해서 기본 버전보다 빠름
# ============================================================
class FasterMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # Q, K, V를 한 번에 프로젝션 (크기 = all_head_size * 3)
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # (batch, seq_len, hidden) → (batch, seq_len, all_head_size * 3)
        qkv = self.qkv_projection(x)

        # 마지막 차원을 3등분해서 Q, K, V 분리
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        batch_size, sequence_length, _ = query.size()

        # 헤드 차원 추가 후 transpose: (batch, num_heads, seq_len, head_size)
        # 헤드별 연산을 배치로 처리하기 위한 reshape
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key   = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 스케일드 닷-프로덕트 어텐션 (기본 버전과 동일 로직)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)

        # (batch, num_heads, seq_len, head_size) → (batch, seq_len, all_head_size)
        # contiguous(): transpose 후 메모리 레이아웃을 연속적으로 만들어야 view 가능
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.all_head_size
        )

        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


# ============================================================
# MLP (Feed-Forward Network)
# 트랜스포머 블록에서 어텐션 다음에 오는 2층 완전연결 네트워크
# hidden → intermediate(4배 확장) → hidden 구조
# ============================================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hidden_size → intermediate_size (보통 4배 확장)
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        # intermediate_size → hidden_size (다시 원래 크기로 압축)
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)    # 차원 확장
        x = self.activation(x) # 비선형성 추가
        x = self.dense_2(x)    # 차원 복원
        x = self.dropout(x)
        return x


# ============================================================
# 트랜스포머 블록 (인코더 레이어 1개)
# Pre-LN 구조: LayerNorm → Attention → 잔차연결 → LayerNorm → MLP → 잔차연결
# (원논문은 Post-LN이지만 Pre-LN이 학습 안정성이 더 좋아 많이 쓰임)
# ============================================================
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # config에 use_faster_attention 없으면 기본값 False
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)

        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])  # 어텐션 전 정규화
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])  # MLP 전 정규화

    def forward(self, x, output_attentions=False):
        # [1] 셀프 어텐션 (Pre-LN: 입력에 먼저 LayerNorm 적용)
        attention_output, attention_probs = self.attention(
            self.layernorm_1(x), output_attentions=output_attentions
        )
        # 잔차 연결: 입력을 그대로 더함 → 기울기 소실 방지, 학습 안정화
        x = x + attention_output

        # [2] MLP (Feed-Forward)
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output  # 잔차 연결

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


# ============================================================
# 트랜스포머 인코더
# Block을 num_hidden_layers개 쌓은 것
# ============================================================
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # nn.ModuleList: 파이썬 리스트와 달리 파라미터가 모델에 등록됨
        self.blocks = nn.ModuleList([
            Block(config) for _ in range(config["num_hidden_layers"])
        ])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        # 각 블록을 순서대로 통과
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)  # 레이어별 어텐션 맵 수집

        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


# ============================================================
# ViT 분류 모델 (최상위 클래스)
# 전체 파이프라인: 이미지 → Embedding → Encoder → [CLS] 토큰 → 분류
# ============================================================
class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        self.embedding  = Embeddings(config)              # 이미지 → 패치 시퀀스
        self.encoder    = Encoder(config)                 # 트랜스포머 인코더
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)  # 최종 분류 헤드

        # 모든 서브모듈에 가중치 초기화 적용
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)

        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
        )

        # [CLS] 토큰(인덱스 0)만 꺼내서 분류에 사용
        # 나머지 패치 토큰들은 버림
        logits = self.classifier(encoder_output[:, 0, :])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        # Linear, Conv2d: 정규분포로 가중치 초기화, bias는 0으로
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # LayerNorm: bias=0, weight=1로 초기화 (항등 변환 상태에서 시작)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 위치 임베딩과 CLS 토큰: truncated normal로 초기화
        # trunc_normal: 너무 크거나 작은 값이 나오지 않도록 분포를 자름
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0, std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0, std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)
