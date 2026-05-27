# ViT(Vision Transformer) 모델의 핵심 구현 파일입니다.
# 이미지 -> 패치 임베딩 -> Transformer Encoder -> 분류 로짓 흐름을 정의합니다.
import math
import torch
from torch import nn


# GELU는 Transformer 계열 모델에서 자주 쓰이는 비선형 활성화 함수입니다.
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

        # 입력 텐서의 각 원소에 GELU 근사식을 적용합니다.
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# 이미지를 작은 패치 단위로 자르고, 각 패치를 hidden_size 차원의 벡터로 바꾸는 단계입니다.
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        # config에서 입력 이미지 크기, 패치 크기, 채널 수, 임베딩 차원을 읽어옵니다.
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        # Conv2d의 kernel_size와 stride를 patch_size로 두면 겹치지 않는 패치 추출과 선형 투영을 한 번에 수행합니다.
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        # 출력 shape: (batch_size, hidden_size, 패치 세로 개수, 패치 가로 개수)
        x = self.projection(x)
        # flatten(2)로 공간 차원을 하나로 합치고, transpose로 (batch, num_patches, hidden_size) 형태를 만듭니다.
        x = x.flatten(2).transpose(1, 2)
        return x


# 패치 임베딩에 CLS 토큰과 위치 임베딩을 더해 Transformer 입력 시퀀스를 만듭니다.
class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 먼저 이미지를 패치 벡터들의 시퀀스로 변환하는 모듈을 준비합니다.
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        # nn.Parameter이므로 학습 과정에서 CLS 토큰 값도 함께 업데이트됩니다.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            # 각 패치 위치와 CLS 위치를 구분할 수 있도록 학습 가능한 위치 정보를 둡니다.
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # 이미지 배치를 패치 임베딩 시퀀스로 변환합니다.
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Transformer는 순서/위치를 직접 알 수 없으므로 위치 임베딩을 더해줍니다.
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


# 하나의 attention head입니다. Q, K, V를 만들어 토큰 간 관계를 계산합니다.
class AttentionHead(nn.Module):
    """
    A single attention head.
    This module is used in the MultiHeadAttention module.

    """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Create the query, key, and value projection layers
        # 같은 입력 x에서 Query, Key, Value를 각각 다른 선형층으로 투영합니다.
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Project the input into query, key, and value
        # The same input is used to generate the query, key, and value,
        # so it's usually called self-attention.
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, attention_head_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        # Query와 Key의 내적으로 토큰끼리 얼마나 관련 있는지 점수를 계산합니다.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # 차원이 클수록 내적 값이 커지는 문제를 줄이기 위해 sqrt(head_size)로 스케일링합니다.
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 마지막 차원 기준 softmax를 적용해 각 토큰이 다른 토큰을 보는 비율로 바꿉니다.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        # attention 확률로 Value를 가중합하여 문맥이 반영된 토큰 표현을 만듭니다.
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)


# 여러 개의 attention head를 병렬로 사용해 다양한 관점의 관계를 학습합니다.
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    This module is used in the TransformerEncoder module.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        # 각 head가 담당할 차원입니다. hidden_size가 head 수로 나누어떨어져야 합니다.
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a list of attention heads
        # head를 ModuleList에 넣어야 PyTorch가 하위 모듈 파라미터를 추적합니다.
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        # 각 head를 독립적으로 실행합니다. 구현은 직관적이지만 FasterMultiHeadAttention보다 느릴 수 있습니다.
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        # 여러 head의 결과를 hidden dimension 방향으로 이어 붙입니다.
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)


# Q/K/V를 한 번의 Linear 연산으로 만들고 head 차원으로 reshape하는 최적화 버전입니다.
class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention module with some optimizations.
    All the heads are processed simultaneously with merged query, key, and value projections.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # The attention head size is the hidden size divided by the number of attention heads
        # 각 head가 담당할 차원입니다. hidden_size가 head 수로 나누어떨어져야 합니다.
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether or not to use bias in the query, key, and value projection layers
        self.qkv_bias = config["qkv_bias"]
        # Create a linear layer to project the query, key, and value
        # 출력 차원을 3배로 만들어 Query, Key, Value를 한 번에 생성합니다.
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        # Project the query, key, and value
        # (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, all_head_size * 3)
        # 한 번의 행렬곱으로 Q/K/V 전체를 계산해 연산 호출 수를 줄입니다.
        qkv = self.qkv_projection(x)
        # Split the projected query, key, and value into query, key, and value
        # (batch_size, sequence_length, all_head_size * 3) -> (batch_size, sequence_length, all_head_size)
        # 마지막 차원을 세 덩어리로 나누어 Query, Key, Value로 분리합니다.
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        batch_size, sequence_length, _ = query.size()
        # shape을 (batch, heads, seq_len, head_dim)으로 바꿔 head별 attention을 한 번에 계산합니다.
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        # Calculate the attention scores
        # softmax(Q*K.T/sqrt(head_size))*V
        # Query와 Key의 내적으로 토큰끼리 얼마나 관련 있는지 점수를 계산합니다.
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        # 차원이 클수록 내적 값이 커지는 문제를 줄이기 위해 sqrt(head_size)로 스케일링합니다.
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 마지막 차원 기준 softmax를 적용해 각 토큰이 다른 토큰을 보는 비율로 바꿉니다.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # Calculate the attention output
        # attention 확률로 Value를 가중합하여 문맥이 반영된 토큰 표현을 만듭니다.
        attention_output = torch.matmul(attention_probs, value)
        # Resize the attention output
        # from (batch_size, num_attention_heads, sequence_length, attention_head_size)
        # To (batch_size, sequence_length, all_head_size)
        # head 차원을 다시 sequence 차원 뒤로 옮겨 원래 hidden 벡터 형태로 복원합니다.
        attention_output = attention_output.transpose(1, 2) \
                                           .contiguous() \
                                           .view(batch_size, sequence_length, self.all_head_size)
        # Project the attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        if not output_attentions:
            return (attention_output, None)
        else:
            return (attention_output, attention_probs)


# Transformer 블록 안의 Feed Forward Network입니다. 각 토큰에 독립적으로 적용됩니다.
class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        # 보통 intermediate_size는 hidden_size의 4배로 두어 표현력을 키웁니다.
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        # hidden_size -> intermediate_size로 확장합니다.
        x = self.dense_1(x)
        x = self.activation(x)
        # intermediate_size -> hidden_size로 다시 축소합니다.
        x = self.dense_2(x)
        x = self.dropout(x)
        return x


# LayerNorm + Self-Attention + Skip Connection + MLP로 구성된 Transformer Encoder 블록입니다.
class Block(nn.Module):
    """
    A single transformer block.
    """

    def __init__(self, config):
        super().__init__()
        # 설정값에 따라 직관적인 구현 또는 빠른 구현 중 하나를 선택합니다.
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        # Pre-LN 구조: attention/MLP에 넣기 전에 LayerNorm을 적용합니다. 학습 안정성에 유리합니다.
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        # Self-attention
        # 정규화된 입력에 self-attention을 적용합니다.
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        # residual connection: 원래 입력 정보를 보존하면서 attention 결과를 더합니다.
        x = x + attention_output
        # Feed-forward network
        # attention을 거친 표현에 다시 정규화 후 MLP를 적용합니다.
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)


# 여러 Transformer Block을 순서대로 쌓은 인코더입니다.
class Encoder(nn.Module):
    """
    The transformer encoder module.
    """

    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        # num_hidden_layers 개수만큼 같은 구조의 블록을 쌓습니다.
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block
        all_attentions = []
        # 각 블록을 통과하면서 토큰 표현이 점점 풍부해집니다.
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        # Return the encoder's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


# CIFAR-10 같은 이미지 분류를 위한 ViT 전체 모델입니다.
class ViTForClassfication(nn.Module):
    """
    The ViT model for classification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # config에서 입력 이미지 크기, 패치 크기, 채널 수, 임베딩 차원을 읽어옵니다.
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        # Create the embedding module
        # 1단계: 이미지를 Transformer가 읽을 수 있는 토큰 시퀀스로 변환합니다.
        self.embedding = Embeddings(config)
        # Create the transformer encoder module
        # 2단계: 토큰 간 attention으로 이미지 전체의 관계를 학습합니다.
        self.encoder = Encoder(config)
        # Create a linear layer to project the encoder's output to the number of classes
        # 3단계: CLS 토큰 표현을 클래스 개수만큼의 점수(logits)로 변환합니다.
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        # 입력 이미지 -> CLS 포함 패치 토큰 시퀀스
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        # Transformer Encoder를 통과한 최종 토큰 표현을 얻습니다.
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        # 0번째 토큰은 CLS 토큰이며, 이미지 전체를 대표하는 분류용 표현으로 사용합니다.
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
    
        # 모델 전체에 apply()로 호출되어 Linear/Conv/LayerNorm/Embedding 초기화를 통일합니다.
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Transformer류 모델에서 흔히 쓰는 작은 표준편차의 정규분포 초기화입니다.
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 위치 임베딩과 CLS 토큰은 trunc_normal로 초기화해 극단값을 줄입니다.
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
