# Vision Transformer 모델 구현에 필요한 라이브러리 불러오기
import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    GELU 활성화 함수를 구현한 클래스

    BERT와 GPT 계열 모델에서 자주 사용되는 GELU 근사식을 사용함
    ReLU처럼 단순히 음수를 0으로 자르는 것이 아니라, 입력값을 부드럽게 조절하는 활성화 함수
    """

    def forward(self, input):
        # GELU 근사 공식 적용
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    이미지를 작은 patch 단위로 나누고, 각 patch를 hidden_size 차원의 벡터로 변환하는 클래스
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # 이미지 전체가 몇 개의 patch로 나뉘는지 계산
        # 예: image_size=32, patch_size=4이면 한 변에 8개, 전체 8*8=64개 patch
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Conv2d를 이용해 patch 분할과 선형 투영을 동시에 수행
        # kernel_size와 stride를 patch_size로 설정하면 겹치지 않는 patch 단위로 이미지를 읽음
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # 입력 형태: (batch_size, num_channels, image_size, image_size)
        # projection 후 형태: (batch_size, hidden_size, patch_grid, patch_grid)
        x = self.projection(x)

        # flatten(2): patch_grid x patch_grid 부분을 하나의 patch sequence로 펼침
        # transpose(1, 2): (batch_size, num_patches, hidden_size) 형태로 변경
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    patch embedding에 CLS token과 position embedding을 더해 Transformer 입력 형태로 만드는 클래스
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # 학습 가능한 CLS token 생성
        # BERT처럼 sequence 맨 앞에 붙이고, 최종 분류에서는 이 token의 출력값을 사용
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # 위치 정보를 담는 position embedding 생성
        # patch token 개수에 CLS token 1개를 더한 길이만큼 필요
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))

        # embedding 출력에 적용할 dropout
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # 이미지를 patch embedding으로 변환
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()

        # CLS token을 batch size만큼 복제
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # CLS token을 patch token sequence 맨 앞에 붙임
        # 결과 sequence 길이: num_patches + 1
        x = torch.cat((cls_tokens, x), dim=1)

        # 각 token에 위치 정보를 더함
        x = x + self.position_embeddings

        # dropout 적용
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    하나의 self-attention head를 구현한 클래스

    Multi-head Attention은 이런 AttentionHead 여러 개를 병렬로 사용함
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size

        # 입력 token을 Query, Key, Value로 변환하는 선형층
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        # attention probability에 적용할 dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 같은 입력 x에서 Query, Key, Value를 만들기 때문에 self-attention이라고 부름
        # 입력 형태: (batch_size, sequence_length, hidden_size)
        # 출력 형태: (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Query와 Key의 내적으로 token 간 유사도 점수 계산
        attention_scores = torch.matmul(query, key.transpose(-1, -2))

        # attention 값이 너무 커지는 것을 방지하기 위해 head 크기의 제곱근으로 나눔
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # softmax로 각 token이 다른 token에 얼마나 집중할지 확률값으로 변환
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # attention 확률을 Value에 곱해 최종 attention 출력 계산
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    여러 개의 AttentionHead를 병렬로 사용한 Multi-head Attention 클래스
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # 각 head가 담당할 벡터 차원 계산
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value 선형층에 bias를 사용할지 여부
        self.qkv_bias = config["qkv_bias"]

        # attention head들을 ModuleList에 저장
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)

        # 여러 head의 출력을 concat한 뒤 다시 hidden_size 차원으로 변환하는 선형층
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 각 attention head에 같은 입력을 넣어 head별 attention 결과 계산
        attention_outputs = [head(x) for head in self.heads]

        # head별 attention output을 마지막 차원 기준으로 이어붙임
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)

        # concat된 출력을 다시 hidden_size 차원으로 투영
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        # attention 확률값이 필요 없는 경우 출력만 반환
        if not output_attentions:
            return (attention_output, None)
        else:
            # 시각화 등을 위해 각 head의 attention probability도 함께 반환
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    Query, Key, Value 계산을 하나의 선형층으로 합쳐 더 빠르게 처리하는 Multi-head Attention 클래스
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        # 각 head가 담당할 차원 계산
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Query, Key, Value 선형층에 bias를 사용할지 여부
        self.qkv_bias = config["qkv_bias"]

        # Query, Key, Value를 한 번에 계산하기 위해 출력 차원을 3배로 설정
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])

        # attention 결과를 다시 hidden_size 차원으로 변환하는 선형층
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 입력을 한 번에 Query, Key, Value용 벡터로 변환
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        qkv = self.qkv_projection(x)

        # 마지막 차원을 3개로 나누어 Query, Key, Value 분리
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        # Query, Key, Value를 head 단위로 나누기 위해 형태 변경
        # (batch_size, sequence_length, all_head_size)
        # -> (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Query와 Key의 내적으로 attention score 계산
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # softmax를 적용해 attention 확률 계산
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # attention 확률을 Value에 곱해 attention output 계산
        attention_output = torch.matmul(attention_probs, value)

        # head별 결과를 다시 하나의 벡터로 합치기 위해 형태 변경
        # (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # -> (batch_size, sequence_length, all_head_size)
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)

        # attention 결과를 hidden_size 차원으로 변환
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        # attention probability 반환 여부 선택
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    Transformer block 내부의 Feed-Forward Network 역할을 하는 MLP 클래스
    """

    def __init__(self, config):
        super().__init__()

        # 첫 번째 선형층: hidden_size -> intermediate_size
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])

        # GELU 활성화 함수
        self.activation = NewGELUActivation()

        # 두 번째 선형층: intermediate_size -> hidden_size
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])

        # dropout 적용
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # 선형층 -> GELU -> 선형층 -> dropout 순서로 처리
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    하나의 Transformer Encoder Block을 구현한 클래스

    구조:
    LayerNorm -> Multi-head Attention -> Skip Connection -> LayerNorm -> MLP -> Skip Connection
    """

    def __init__(self, config):
        super().__init__()

        # 설정값에 따라 일반 MultiHeadAttention 또는 최적화된 FasterMultiHeadAttention 사용
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)

        # attention 앞에 적용하는 LayerNorm
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])

        # feed-forward network 역할의 MLP
        self.mlp = MLP(config)

        # MLP 앞에 적용하는 LayerNorm
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # LayerNorm 후 self-attention 수행
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)

        # 원래 입력 x와 attention 출력을 더하는 skip connection
        x = x + attention_output

        # LayerNorm 후 MLP 수행
        mlp_output = self.mlp(self.layernorm_2(x))

        # 원래 입력 x와 MLP 출력을 더하는 skip connection
        x = x + mlp_output

        # attention probability 반환 여부 선택
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    여러 개의 Transformer Encoder Block을 쌓은 Encoder 클래스
    """

    def __init__(self, config):
        super().__init__()

        # num_hidden_layers 개수만큼 Transformer block 생성
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # 각 block을 순서대로 통과시키며 token representation 업데이트
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)

        # attention map 반환 여부 선택
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    """
    이미지 분류를 위한 Vision Transformer 모델 클래스
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        # 이미지 patch embedding, CLS token, position embedding 생성 모듈
        self.embedding = Embeddings(config)

        # Transformer Encoder 모듈
        self.encoder = Encoder(config)

        # CLS token의 최종 출력을 클래스 개수만큼의 logit으로 변환하는 분류기
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # 모델 전체 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # 입력 이미지를 Transformer가 처리할 수 있는 token sequence로 변환
        embedding_output = self.embedding(x)

        # Transformer Encoder 통과
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)

        # 0번째 token인 CLS token의 출력만 사용해 이미지 클래스 예측
        logits = self.classifier(encoder_output[:, 0, :])

        # attention map 반환 여부 선택
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        """
        모델 내부 모듈의 가중치를 초기화하는 함수
        """
        # Linear와 Conv2d의 weight는 정규분포로 초기화하고, bias는 0으로 초기화
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        # LayerNorm의 bias는 0, weight는 1로 초기화
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Embeddings 내부의 position embedding과 CLS token은 truncated normal로 초기화
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
