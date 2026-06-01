import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    GELU 활성화 함수 구현.
    논문: https://arxiv.org/abs/1606.08415

    ReLU와 달리 입력값을 확률적으로 통과시키는 방식으로,
    Transformer 계열 모델에서 표준으로 사용된다.
    수식: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    이미지를 패치로 분할하고 벡터 공간으로 projection하는 모듈.

    ViT의 핵심 아이디어: 이미지를 NLP의 토큰처럼 취급하기 위해
    P×P 크기의 패치로 잘라서 각 패치를 하나의 벡터로 변환한다.

    Conv2d를 사용하는 이유:
      - kernel_size=patch_size, stride=patch_size로 설정하면
        겹치지 않는 패치를 한 번에 추출 + linear projection까지 수행 가능
      - 별도의 reshape 없이 효율적으로 처리
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # 전체 패치 수 계산: (이미지 크기 / 패치 크기)^2
        # 예: 32x32 이미지, patch_size=4 → (32/4)^2 = 64개 패치
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Conv2d로 패치 추출 + hidden_size 차원으로 projection을 한 번에 수행
        # kernel_size = stride = patch_size이므로 패치 간 겹침 없음
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # 입력: (batch_size, num_channels, image_size, image_size)
        # Conv2d 통과 후: (batch_size, hidden_size, H/P, W/P)
        x = self.projection(x)
        # flatten(2): 공간 차원(H/P, W/P)을 하나로 펼침 → (batch_size, hidden_size, num_patches)
        # transpose(1, 2): 차원 순서 변경 → (batch_size, num_patches, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    패치 임베딩 + [CLS] 토큰 + Position Embedding을 합치는 모듈.

    최종 출력 shape: (batch_size, num_patches + 1, hidden_size)
      - +1은 [CLS] 토큰

    [CLS] 토큰이란?
      BERT에서 도입된 방식. 시퀀스 앞에 학습 가능한 토큰을 추가하고,
      모든 패치의 정보를 이 토큰이 집약하도록 학습시킨다.
      최종 분류는 이 [CLS] 토큰의 출력만 사용한다.

    Position Embedding이란?
      Transformer는 순서 정보가 없으므로, 각 패치의 위치 정보를
      학습 가능한 벡터로 추가해준다. 논문에서는 1D position embedding 사용.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # [CLS] 토큰: 학습 가능한 파라미터 (1, 1, hidden_size)
        # 배치 내 모든 샘플에 동일하게 추가되므로 batch 차원은 1로 초기화 후 expand
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # Position embedding: 패치 수 + 1 ([CLS] 토큰 포함)
        # 각 위치마다 고유한 학습 가능한 벡터를 더해 순서 정보를 주입
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # 1. 패치 임베딩: (batch_size, num_patches, hidden_size)
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # 2. [CLS] 토큰을 배치 크기에 맞게 복사
        # (1, 1, hidden_size) → (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 3. [CLS] 토큰을 시퀀스 맨 앞에 붙임
        # (batch_size, num_patches+1, hidden_size)
        x = torch.cat((cls_tokens, x), dim=1)

        # 4. Position embedding 더하기 (element-wise addition)
        x = x + self.position_embeddings

        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    단일 Attention Head 구현.

    Self-Attention 계산 과정:
      1. 입력 x에서 Query(Q), Key(K), Value(V)를 각각 Linear projection으로 생성
      2. Attention score = Q * K^T / sqrt(head_size)  ← sqrt로 나눠 gradient 안정화
      3. Softmax로 정규화 → Attention weight (각 패치 간 관계 강도)
      4. Attention output = Attention weight * V

    "Self"-Attention인 이유: Q, K, V 모두 같은 입력 x에서 생성되기 때문.
    """

    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # Q, K, V 각각을 hidden_size → attention_head_size 차원으로 projection
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key   = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 입력 x → Q, K, V 생성
        # (batch_size, seq_len, hidden_size) → (batch_size, seq_len, attention_head_size)
        query = self.query(x)
        key   = self.key(x)
        value = self.value(x)

        # Attention score 계산: Q * K^T
        # (batch_size, seq_len, head_size) x (batch_size, head_size, seq_len)
        # → (batch_size, seq_len, seq_len): 각 토큰 쌍 간의 유사도
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # head_size의 sqrt로 나눠 softmax 입력값의 분산을 안정화 (논문 수식 그대로)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Softmax: 각 토큰이 다른 토큰에 얼마나 집중할지 확률로 변환
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Attention output: 가중합으로 각 토큰의 새로운 표현 계산
        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, head_size)
        # → (batch_size, seq_len, head_size)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 구현 (기본 버전).

    여러 개의 AttentionHead를 병렬로 실행하고 출력을 이어붙인다.

    Multi-Head를 쓰는 이유:
      각 head가 서로 다른 관점(예: 모양, 색상, 위치 등)에서
      패치 간 관계를 학습할 수 있어 표현력이 높아진다.

    이 버전은 head를 for loop으로 개별 실행하므로 직관적이지만
    FasterMultiHeadAttention 대비 속도가 느리다.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # 각 head의 차원: hidden_size를 head 수로 균등 분할
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # 각 head를 독립적인 모듈로 생성 (nn.ModuleList로 관리)
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)

        # 각 head의 출력을 concat한 후 다시 hidden_size로 projection
        # all_head_size = num_heads * head_size = hidden_size (대부분의 경우)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 head별로 attention 계산 (리스트 컴프리헨션)
        attention_outputs = [head(x) for head in self.heads]

        # 각 head의 attention output만 추출해 마지막 차원(head_size)으로 이어붙임
        # (batch_size, seq_len, head_size) * num_heads → (batch_size, seq_len, all_head_size)
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)

        # all_head_size → hidden_size로 projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            # 시각화용: 모든 head의 attention prob을 쌓아 반환
            # (batch_size, num_heads, seq_len, seq_len)
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 최적화 버전.

    기본 버전(MultiHeadAttention)과 수학적으로 동일하지만,
    Q/K/V를 head별로 따로 계산하는 대신 하나의 큰 Linear layer로
    한 번에 projection한 뒤 chunk로 분할하여 병렬 처리한다.

    핵심 최적화:
      - qkv_projection: hidden_size → all_head_size * 3 (Q+K+V를 한 번에)
      - reshape + transpose로 head 차원을 분리해 배치 행렬 곱으로 처리
      → GPU에서 연산을 더 효율적으로 병렬화 가능
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]

        # Q, K, V를 한 번에 계산: hidden_size → all_head_size * 3
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Q, K, V를 한 번에 projection
        # (batch_size, seq_len, hidden_size) → (batch_size, seq_len, all_head_size * 3)
        qkv = self.qkv_projection(x)

        # 마지막 차원을 3등분하여 Q, K, V로 분리
        # 각각의 shape: (batch_size, seq_len, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        # head 차원을 분리하고 배치 행렬 곱을 위해 transpose
        # (batch_size, seq_len, all_head_size)
        # → (batch_size, seq_len, num_heads, head_size)
        # → (batch_size, num_heads, seq_len, head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key   = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Attention score: Q * K^T / sqrt(head_size)
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # Attention output: attention_probs * V
        # (batch_size, num_heads, seq_len, head_size)
        attention_output = torch.matmul(attention_probs, value)

        # head 차원을 다시 합침
        # (batch_size, num_heads, seq_len, head_size)
        # → (batch_size, seq_len, num_heads, head_size)
        # → (batch_size, seq_len, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)

        # all_head_size → hidden_size로 projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    Transformer Block 내부의 Feed-Forward Network (FFN).

    구조: Linear → GELU → Linear → Dropout
    논문에서는 intermediate_size = 4 * hidden_size로 설정.

    역할:
      Self-Attention이 토큰 간 관계를 모델링한다면,
      MLP는 각 토큰의 표현을 개별적으로 변환한다.
      (position-wise, 즉 각 토큰에 독립적으로 적용)
    """

    def __init__(self, config):
        super().__init__()
        # hidden_size → intermediate_size (4배 확장)로 표현력 증가
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        # intermediate_size → hidden_size로 다시 축소
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    단일 Transformer Encoder Block.

    구조 (Pre-LN 방식):
      x → LayerNorm → Multi-Head Attention → + (residual) → x
      x → LayerNorm → MLP → + (residual) → x

    Pre-LN vs Post-LN:
      원래 논문(Attention is All You Need)은 Post-LN을 사용하지만,
      ViT 논문은 안정적인 학습을 위해 Pre-LN(Attention 전에 LN 적용)을 사용한다.

    Residual Connection(Skip Connection):
      그래디언트가 LayerNorm/Attention을 건너뛰어 직접 흐를 수 있게 하여
      깊은 네트워크에서도 학습이 안정적으로 이루어진다.
    """

    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        # config에 따라 최적화 버전 또는 기본 버전 선택
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)

        # Pre-LN: Attention 전후 각각 LayerNorm 적용
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # 1. Pre-LN → Self-Attention → Residual Connection
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attention_output  # skip connection

        # 2. Pre-LN → MLP → Residual Connection
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output  # skip connection

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    Transformer Encoder: Block을 num_hidden_layers번 쌓은 구조.

    각 Block의 출력이 다음 Block의 입력으로 전달되며,
    레이어가 깊어질수록 더 추상적인 특징을 학습한다.

    output_attentions=True이면 모든 Block의 attention map을 반환.
    (시각화 목적: utils.py의 visualize_attention에서 사용)
    """

    def __init__(self, config):
        super().__init__()
        # num_hidden_layers개의 Block을 nn.ModuleList로 관리
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        if not output_attentions:
            return (x, None)
        else:
            # 모든 레이어의 attention prob 리스트 반환
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    ViT 전체 분류 모델.

    전체 흐름:
      이미지
        → PatchEmbeddings: 패치 분할 + projection
        → Embeddings: [CLS] 토큰 추가 + position embedding
        → Encoder: L개의 Transformer Block
        → classifier: [CLS] 토큰의 출력 → 클래스 수로 projection
        → logits (분류 결과)

    분류에 [CLS] 토큰만 사용하는 이유:
      학습 과정에서 [CLS] 토큰이 모든 패치의 정보를
      Self-Attention을 통해 집약하도록 학습되기 때문.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        # 모델 구성 요소 초기화
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        # [CLS] 토큰 출력(hidden_size) → 클래스 수로 projection
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # 가중치 초기화 (논문 방식: truncated normal)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # 1. 패치 임베딩 + [CLS] 토큰 + position embedding
        embedding_output = self.embedding(x)

        # 2. Transformer Encoder 통과
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)

        # 3. [CLS] 토큰(index 0)의 출력만 추출하여 분류
        # encoder_output: (batch_size, seq_len, hidden_size)
        # encoder_output[:, 0, :]: (batch_size, hidden_size)
        logits = self.classifier(encoder_output[:, 0, :])

        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        """
        가중치 초기화 함수.

        - Linear, Conv2d: 평균 0, 표준편차 initializer_range의 정규분포로 초기화
          bias는 0으로 초기화
        - LayerNorm: weight=1, bias=0으로 초기화 (항등 변환 시작)
        - Embeddings: position embedding과 cls_token은
          truncated normal로 초기화 (dtype 보존을 위해 float32로 변환 후 복원)
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)