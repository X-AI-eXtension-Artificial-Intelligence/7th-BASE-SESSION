"""
Vision Transformer (ViT) 구현체.

논문: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR 2021)
구조 흐름: 이미지 → 패치 시퀀스 → [CLS] + 위치 임베딩 → Transformer 인코더 → [CLS] 토큰으로 분류
"""

import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    GELU 활성화 함수 (Gaussian Error Linear Unit).
    BERT/GPT 등에서 표준으로 쓰이는 형태이며, 원본 ViT 논문도 GELU 사용.
    ReLU보다 부드러운 비선형성을 제공해 학습 안정성이 좋다고 알려져 있음.
    """

    def forward(self, input):
        # GELU의 tanh 근사: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    이미지를 패치로 분할하고 각 패치를 벡터 공간으로 사영.

    핵심 트릭: 패치 자르기 + 선형 사영을 Conv2d 한 번으로 처리.
    Conv2d(kernel=patch_size, stride=patch_size)는 겹치지 않는 패치마다
    내적을 계산하므로, "각 패치를 flatten 후 Linear에 통과"와 수학적으로 동일.

    Shape 변환:
        (B, C, H, W) -> (B, num_patches, hidden_size)
        예: (B, 3, 32, 32) -> (B, 64, 48)   [32/4=8, 8×8=64개 패치]
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]          # 입력 이미지 한 변 크기 (32)
        self.patch_size = config["patch_size"]          # 패치 한 변 크기 (4)
        self.num_channels = config["num_channels"]      # 입력 채널 수 (3, RGB)
        self.hidden_size = config["hidden_size"]        # 패치 임베딩 차원 (48)
        # 한 변에 image_size/patch_size 개 패치 → 전체 패치 수는 그 제곱
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 64
        # Conv2d로 패치 추출 + 선형 사영을 동시에 수행
        # kernel과 stride가 동일하므로 패치끼리 겹치지 않음
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (B, 3, 32, 32) -> Conv2d(stride=4) -> (B, 48, 8, 8)
        x = self.projection(x)
        # 공간 차원 2개를 하나로 펴고 transpose해서 시퀀스 형태로
        # (B, 48, 8, 8) -> flatten(2) -> (B, 48, 64) -> transpose -> (B, 64, 48)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    패치 임베딩에 [CLS] 토큰과 위치 임베딩을 결합.
    논문 식 (1) z_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos 에 해당.

    Shape 변환:
        (B, num_patches, hidden) -> (B, num_patches+1, hidden)
        예: (B, 64, 48) -> (B, 65, 48)   [CLS 토큰 1개 prepend]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # 학습 가능한 [CLS] 토큰. BERT처럼 시퀀스 맨 앞에 붙어서 분류에 사용됨.
        # 모든 배치가 공유하는 단일 벡터이므로 (1, 1, hidden_size)로 시작
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # 학습 가능한 위치 임베딩. CLS 토큰 자리까지 포함해 num_patches+1개.
        # 1D 임베딩(원본 ViT 논문 D.4에서 2D-aware보다 차이 없다고 결론)
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # 1) 패치 임베딩: (B, 3, 32, 32) -> (B, 64, 48)
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # 2) [CLS] 토큰을 배치 크기만큼 복제: (1, 1, 48) -> (B, 1, 48)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 3) CLS 토큰을 시퀀스 맨 앞에 붙이기: (B, 64, 48) -> (B, 65, 48)
        x = torch.cat((cls_tokens, x), dim=1)
        # 4) 위치 임베딩 더하기 (broadcasting: (1, 65, 48) + (B, 65, 48))
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    단일 어텐션 헤드 (학습용 직관 버전).
    MultiHeadAttention에서 여러 개를 ModuleList로 쌓아서 사용.

    수식: Attention(Q, K, V) = softmax(QK^T / √d_k) V
    같은 입력 x로부터 Q, K, V를 모두 생성하므로 self-attention.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size  # 한 헤드당 차원 = hidden / num_heads
        # Q, K, V 각각을 위한 선형 사영 레이어
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 입력을 Q, K, V로 각각 사영
        # (B, seq_len, hidden) -> (B, seq_len, head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # 1) Q·K^T로 attention score 계산
        # (B, seq_len, head_size) × (B, head_size, seq_len) -> (B, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # 2) √d_k로 스케일링 (gradient가 너무 커지는 것 방지)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 3) softmax로 확률 분포로 변환
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # 4) attention 확률로 value를 가중합
        # (B, seq_len, seq_len) × (B, seq_len, head_size) -> (B, seq_len, head_size)
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    다중 헤드 어텐션 (직관적 구현, 느림).
    AttentionHead를 num_attention_heads개 만들어서 for문으로 돌림.
    실제 학습에는 아래의 FasterMultiHeadAttention이 사용됨 (config의 use_faster_attention=True).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # 한 헤드당 차원: hidden_size를 head 개수로 나눈 값
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        # 헤드 개수만큼 AttentionHead 모듈 생성
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # 모든 헤드 출력을 concat 후 다시 hidden_size로 사영하는 출력 레이어
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 헤드별로 attention 계산 (병렬화 안 됨, 그래서 느림)
        attention_outputs = [head(x) for head in self.heads]
        # 헤드 출력들을 마지막 차원으로 concat
        # 각 (B, seq, head_size) → concat → (B, seq, num_heads * head_size) = (B, seq, all_head_size)
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # 다시 hidden_size로 사영 (원본 Transformer의 W_O에 해당)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # attention map 출력 옵션 (시각화용)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    다중 헤드 어텐션 (최적화 버전, 실제 학습에 사용).

    핵심 차이: 모든 헤드의 Q, K, V projection을 단일 행렬곱으로 묶음.
    Linear(hidden, all_head_size * 3)으로 한 번에 처리 후 reshape으로 헤드 분리.
    GPU에서 큰 행렬곱 1개가 작은 행렬곱 N개보다 훨씬 효율적.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        # Q, K, V를 한 번에 사영 (출력 차원이 all_head_size * 3)
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 1) QKV 한 번에 사영
        # (B, seq, hidden) -> (B, seq, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # 2) 마지막 차원을 3등분해서 Q, K, V로 분리
        # 각각 (B, seq, all_head_size)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # 3) 헤드 차원 분리를 위해 reshape
        # (B, seq, all_head_size) -> (B, seq, num_heads, head_size) -> (B, num_heads, seq, head_size)
        # transpose하면 num_heads 차원이 앞으로 와서 헤드별로 병렬 attention 가능
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # 4) scaled dot-product attention (헤드별 병렬)
        # (B, num_heads, seq, head_size) × (B, num_heads, head_size, seq) -> (B, num_heads, seq, seq)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # (B, num_heads, seq, seq) × (B, num_heads, seq, head_size) -> (B, num_heads, seq, head_size)
        attention_output = torch.matmul(attention_probs, value)
        # 5) 헤드 차원을 다시 시퀀스 뒤로 보내고 합치기
        # (B, num_heads, seq, head_size) -> (B, seq, num_heads, head_size) -> (B, seq, all_head_size)
        # contiguous()는 view를 위한 메모리 연속화
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # 6) 출력 사영
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    논문 식 (3)의 MLP 부분: hidden -> intermediate(4×hidden) -> hidden.
    중간에 GELU 비선형성. 시퀀스의 각 토큰에 독립적으로 적용됨.
    """

    def __init__(self, config):
        super().__init__()
        # 첫 번째 dense layer: 차원을 4배로 확장 (hidden=48 -> intermediate=192)
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        # 두 번째 dense layer: 다시 hidden 차원으로 축소 (192 -> 48)
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)         # 차원 확장
        x = self.activation(x)      # GELU
        x = self.dense_2(x)         # 차원 축소
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    하나의 Transformer 블록.
    논문 식 (2), (3)을 합친 형태:
        z' = MSA(LN(z)) + z          # 식 (2)
        z  = MLP(LN(z')) + z'        # 식 (3)

    Pre-LN 구조 (ViT/GPT 표준): LayerNorm을 residual 안이 아니라 앞에 둔다.
    원본 Transformer의 Post-LN보다 학습 안정성이 좋다고 알려져 있음.
    """

    def __init__(self, config):
        super().__init__()
        # config에 따라 직관 버전 또는 최적화 버전 선택
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        # Attention 전에 적용할 LN
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        # MLP 전에 적용할 LN
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention: LN -> Attention -> Skip
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + attention_output  # residual connection
        # FFN: LN -> MLP -> Skip
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x + mlp_output        # residual connection
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    Transformer 인코더 = Block을 num_hidden_layers번 직렬로 쌓은 것.
    여기 CIFAR-10용 config에서는 4개. 원본 ViT-Base는 12개, ViT-Large는 24개.
    """

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # 각 블록을 순차적으로 통과시키면서 출력 갱신
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    최종 모델: ViT for image classification.

    전체 흐름:
        이미지 (B, 3, 32, 32)
          → embedding: 패치 추출 + CLS + 위치 임베딩 → (B, 65, 48)
          → encoder: Transformer 블록 L번 → (B, 65, 48)
          → CLS 토큰만 추출 (인덱스 0) → (B, 48)
          → classifier (Linear 48→10) → 로짓 (B, 10)

    논문 식 (4): y = LN(z_L^0) 에 해당하는 부분이 CLS 토큰 추출 + classifier.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # 임베딩 + 인코더 + 분류기
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        # 분류 헤드: hidden_size -> num_classes (CIFAR-10이면 10)
        # 원본 논문은 pre-training 시 hidden layer 있는 MLP, fine-tuning 시 단일 linear
        # 여기서는 단순화를 위해 처음부터 단일 linear
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # 1) 임베딩
        embedding_output = self.embedding(x)
        # 2) 인코더
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # 3) CLS 토큰의 최종 representation으로 분류
        # encoder_output: (B, 65, 48), [:, 0, :]은 시퀀스의 첫 번째 토큰(=CLS) 추출 → (B, 48)
        logits = self.classifier(encoder_output[:, 0, :])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def _init_weights(self, module):
        """
        BERT 스타일 가중치 초기화.
        - Linear/Conv2d: 정규분포 (std=0.02)
        - LayerNorm: bias=0, weight=1
        - position_embeddings, cls_token: truncated normal
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            # 위치 임베딩과 CLS 토큰은 truncated normal로 초기화
            # (일반 normal보다 극단값이 제거되어 학습 안정성 ↑)
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
