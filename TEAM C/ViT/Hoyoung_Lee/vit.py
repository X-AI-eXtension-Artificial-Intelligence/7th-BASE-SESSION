import math
import torch
from torch import nn


class NewGELUActivation(nn.Module):
    """
    BERT나 GPT에서 사용되는 GELU 활성화 함수.
    비선형성을 추가하며, ReLU보다 더 부드러운 형태를 가짐.
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    """
    입력 이미지를 여러 개의 작은 패치(조각)로 나누고 각 패치를 벡터로 변환(임베딩)하는 계층.
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        
        # 전체 패치의 개수 계산 = (가로 크기 / 패치 크기) * (세로 크기 / 패치 크기)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Conv2d를 이용해 이미지를 패치 크기만큼 성큼성큼(stride) 건너뛰며 자르고, 동시에 임베딩 차원으로 투영
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # 2D 이미지 특징 맵을 1D 시퀀스로 펼침: (배치, 채널, H, W) -> (배치, 패치 수, 은닉 차원)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    패치 임베딩에 클래스 토큰([CLS])을 추가하고, 각 패치의 위치 정보(Positional Embedding)를 더하는 계층.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        
        # 이미지 전체의 특징을 요약하여 분류에 사용할 학습 가능한 [CLS] 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        
        # 위치 임베딩: 패치의 순서/위치 정보를 모델에 알려주기 위함 ([CLS] 토큰 1개 + 패치 개수만큼의 위치 필요)
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.size(0)
        
        # 배치 크기에 맞게 [CLS] 토큰을 복사하여 확장
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 패치 시퀀스 맨 앞에 [CLS] 토큰 연결
        x = torch.cat((cls_tokens, x), dim=1)
        # 위치 임베딩 정보 더하기
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    """
    단일 어텐션 헤드.
    입력 시퀀스의 각 요소들이 서로 얼마나 연관성이 있는지를 계산.
    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        # Query(질의), Key(키), Value(값)를 생성하는 선형 계층
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.attention_head_size = attention_head_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # 어텐션 스코어 계산: Q와 K의 내적을 통해 연관성 파악 후 스케일링
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Softmax를 통과시켜 확률값으로 변환
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 구한 확률(가중치)을 바탕으로 Value 값들을 가중합
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


class MultiHeadAttention(nn.Module):
    """
    여러 개의 AttentionHead를 병렬로 사용하는 멀티 헤드 어텐션 모듈.
    다양한 관점에서 정보의 연관성을 학습함.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        
        # 여러 개의 헤드 생성
        self.heads = nn.ModuleList([
            AttentionHead(self.hidden_size, self.attention_head_size, config["attention_probs_dropout_prob"], self.qkv_bias)
            for _ in range(self.num_attention_heads)
        ])
        
        # 모든 헤드의 출력을 하나로 합친 뒤 원래 차원으로 복원하는 투영 계층
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # 모든 헤드의 출력을 계산 후 마지막 차원을 기준으로 이어 붙임
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([out for out, _ in attention_outputs], dim=-1)
        
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([probs for _, probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    """
    연산 속도가 최적화된 멀티 헤드 어텐션.
    여러 헤드를 반복문으로 돌리지 않고, Q, K, V 투영을 한 번의 큰 선형 연산으로 병렬 처리함.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        
        # Q, K, V를 동시에 계산하기 위해 출력 크기를 3배로 잡은 선형 계층
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        qkv = self.qkv_projection(x)
        # 결과물을 Q, K, V 3개로 분리
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        
        batch_size, sequence_length, _ = query.size()
        
        # 행렬 연산을 위해 텐서의 형태를 (배치, 헤드 수, 시퀀스 길이, 헤드 차원)으로 변형
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # 어텐션 스코어 및 확률 계산
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # 어텐션 결과 계산 및 형태 복원
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.all_head_size)
        
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


class MLP(nn.Module):
    """
    트랜스포머 내부에 위치한 다층 퍼셉트론(피드포워드 신경망).
    특징들을 비선형적으로 조합하여 복잡한 패턴을 학습.
    """
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"]) # 차원 확장
        self.activation = NewGELUActivation() # 활성화 함수
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"]) # 원래 차원으로 축소
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    트랜스포머의 기본 구성 단위인 단일 블록.
    (LayerNorm -> MultiHeadAttention -> 잔차 연결) + (LayerNorm -> MLP -> 잔차 연결) 구조.
    """
    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        # 설정에 따라 최적화된 어텐션 또는 일반 어텐션 선택
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # 첫 번째 LayerNorm 적용 후 어텐션 통과
        attention_output, attention_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # 잔차(Skip) 연결: 입력값을 출력값에 더해 기울기 소실 방지
        x = x + attention_output
        
        # 두 번째 LayerNorm 적용 후 MLP 통과
        mlp_output = self.mlp(self.layernorm_2(x))
        # 다시 잔차 연결
        x = x + mlp_output
        
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


class Encoder(nn.Module):
    """
    여러 개의 트랜스포머 블록을 쌓아 만든 인코더.
    """
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, x, output_attentions=False):
        all_attentions = []
        # 각 블록을 순차적으로 통과시킴
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
    이미지 분류를 위한 최종 Vision Transformer 모델.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        
        # 모듈 조립: 임베딩 -> 인코더 -> 분류기
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        # 최종 분류 계층: [CLS] 토큰의 출력을 받아 각 클래스에 대한 점수(Logits) 계산
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        self.apply(self._init_weights) # 초기화 함수 적용

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        
        # 시퀀스의 첫 번째 요소인 [CLS] 토큰의 출력(encoder_output[:, 0, :])만 사용하여 분류 진행
        logits = self.classifier(encoder_output[:, 0, :])
        
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
    def _init_weights(self, module):
        """
        신경망 레이어의 가중치를 정규 분포 등으로 초기화하는 헬퍼 함수.
        학습 초기의 안정성과 수렴 속도를 돕습니다.
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