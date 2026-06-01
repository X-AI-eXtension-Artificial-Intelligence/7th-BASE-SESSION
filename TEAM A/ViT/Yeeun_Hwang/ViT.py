import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        # GELU 활성화 함수의 tanh 근사 공식:
        # 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        # ReLU와 달리 음수 입력에서도 부드러운 곡선을 가지며, 트랜스포머 계열 모델에서 널리 사용됨
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # 전체 이미지를 패치 크기로 나눈 뒤 가로/세로 패치 수를 제곱하여 총 패치 수를 계산
        # 예: 224x224 이미지를 16x16 패치로 나누면 (224/16)^2 = 196개의 패치
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Conv2d를 이용해 패치 분할과 hidden_size 차원으로의 선형 투영을 동시에 수행
        # kernel_size와 stride를 모두 patch_size로 설정하여 겹치지 않는 패치를 추출
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        # flatten(2): H, W 두 차원을 하나의 num_patches 차원으로 합침
        # transpose(1, 2): (batch, hidden, patches) -> (batch, patches, hidden) 순서로 변환
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # BERT의 [CLS] 토큰과 동일한 개념으로, 시퀀스 전체를 대표하는 학습 가능한 벡터
        # shape: (1, 1, hidden_size) - 배치 전체에 동일하게 사용되므로 1로 초기화
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # 각 토큰의 위치 정보를 학습하는 파라미터
        # num_patches + 1: 패치 토큰 수에 [CLS] 토큰 1개를 더한 전체 시퀀스 길이
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # expand로 배치 크기만큼 [CLS] 토큰을 확장 (실제 메모리 복사 없이 뷰만 생성)
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # [CLS] 토큰을 시퀀스의 맨 앞에 연결하여 시퀀스 길이가 (num_patches + 1)이 됨
        x = torch.cat((cls_tokens, x), dim=1)
        # 패치 임베딩에 위치 임베딩을 더해 각 토큰에 위치 정보를 부여
        # position_embeddings의 shape (1, seq_len, hidden_size)가 배치 차원으로 브로드캐스팅됨
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # 입력을 Query, Key, Value 공간으로 각각 투영하는 선형 레이어
        # hidden_size -> attention_head_size 차원으로 축소됨
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 동일한 입력 x로부터 Query, Key, Value를 각각 생성 (Self-Attention)
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Scaled Dot-Product Attention: softmax(Q * K^T / sqrt(head_size)) * V
        # key.transpose(-1, -2): (batch, seq, head_size) -> (batch, head_size, seq) 로 전치하여 행렬곱 수행
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # attention_head_size의 제곱근으로 나누어 내적 값이 너무 커지는 것을 방지 (그래디언트 안정화)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 소프트맥스로 어텐션 가중치를 확률 분포로 변환 (각 행의 합이 1)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # 어텐션 가중치와 Value의 가중합으로 최종 어텐션 출력 계산
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # hidden_size를 헤드 수로 균등하게 나누어 각 헤드의 차원을 결정
        # 예: hidden_size=768, num_heads=12 이면 각 헤드는 64차원을 담당
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        # 각 어텐션 헤드를 독립적인 모듈로 생성하여 ModuleList에 등록
        # 각 헤드는 서로 다른 표현 공간에서 어텐션을 학습함
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # 모든 헤드의 출력을 이어붙인 뒤 다시 hidden_size로 투영하는 레이어
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 헤드에 동일한 입력 x를 넣어 어텐션 출력과 확률을 계산
        attention_outputs = [head(x) for head in self.heads]
        # 모든 헤드의 출력을 마지막 차원(feature)으로 이어붙임
        # (batch, seq, head_size) * num_heads -> (batch, seq, all_head_size)
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # 이어붙인 출력을 다시 hidden_size로 선형 투영
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            # 각 헤드의 어텐션 확률을 dim=1에 쌓아 (batch, num_heads, seq, seq) 형태로 반환
            # 어텐션 시각화 등에 활용 가능
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        # Q, K, V 투영을 하나의 Linear로 통합하여 단일 행렬 연산으로 처리
        # hidden_size -> all_head_size * 3 으로 한 번에 투영한 뒤 분리함
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # 마지막 차원을 3등분하여 Query, Key, Value로 분리
        # (batch, seq, all_head_size * 3) -> 각각 (batch, seq, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        # view로 헤드 차원을 분리한 뒤 transpose로 헤드를 batch 바로 뒤로 이동
        # (batch, seq, heads, head_size) -> (batch, heads, seq, head_size)
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # (batch, heads, seq, head_size) x (batch, heads, head_size, seq) -> (batch, heads, seq, seq)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # (batch, heads, seq, seq) x (batch, heads, seq, head_size) -> (batch, heads, seq, head_size)
        attention_output = torch.matmul(attention_probs, value)
        # transpose 후 메모리를 연속적으로 재배열(contiguous)하고
        # 헤드 차원을 다시 합쳐 (batch, seq, all_head_size) 형태로 변환
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        # hidden_size -> intermediate_size 로 차원을 확장 (보통 4배)
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        # intermediate_size -> hidden_size 로 다시 원래 차원으로 축소
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
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        # use_faster_attention 플래그에 따라 어텐션 구현체를 선택
        # FasterMultiHeadAttention은 Q/K/V를 하나의 행렬 연산으로 처리하여 더 효율적
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        # Pre-LN 구조: 어텐션과 MLP 각각의 입력에 LayerNorm을 먼저 적용
        # Post-LN(출력에 적용) 대비 학습 초기 안정성이 높음
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-Attention
        # layernorm_1을 적용한 입력으로 어텐션을 계산하고, 원본 x에 더함 (Residual Connection)
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attention_output
        # Feed-Forward Network
        # layernorm_2를 적용한 입력으로 MLP를 통과하고, 다시 Residual Connection 적용
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # num_hidden_layers 수만큼 트랜스포머 블록을 쌓아 깊은 인코더를 구성
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # 각 블록을 순서대로 통과하며 표현을 점진적으로 정제
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                # 각 레이어의 어텐션 확률을 리스트에 누적 (레이어별 어텐션 분석에 활용)
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # 이미지를 패치로 분할하고 위치 임베딩을 더해 시퀀스 형태로 변환
        self.embedding = Embeddings(config)
        # 멀티헤드 셀프 어텐션과 MLP로 구성된 트랜스포머 블록을 여러 층 쌓은 인코더
        self.encoder = Encoder(config)
        # 인코더 출력의 [CLS] 토큰 벡터를 클래스 수만큼의 로짓으로 변환하는 분류 헤드
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # 모든 서브모듈에 재귀적으로 가중치 초기화를 적용
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # 이미지를 패치 임베딩 + 위치 임베딩 시퀀스로 변환
        embedding_output = self.embedding(x)
        # 트랜스포머 인코더를 통과하여 각 토큰의 문맥적 표현을 계산
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # encoder_output[:, 0, :] 로 [CLS] 토큰(0번째)의 출력만 추출하여 분류에 사용
        # [CLS] 토큰은 전체 시퀀스의 전역 정보를 집약하도록 학습됨
        logits = self.classifier(encoder_output[:, 0, :])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 평균 0, 표준편차 initializer_range인 정규분포로 가중치 초기화
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                # 편향은 0으로 초기화
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm의 스케일(weight)은 1, 이동(bias)은 0으로 초기화하여 항등 변환으로 시작
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            # trunc_normal_: 정규분포에서 ±2σ를 벗어나는 값을 잘라내어 극단적인 초기값 방지
            # fp16/bf16 환경에서의 수치 안정성을 위해 float32로 변환 후 초기화하고 원래 dtype으로 복원
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