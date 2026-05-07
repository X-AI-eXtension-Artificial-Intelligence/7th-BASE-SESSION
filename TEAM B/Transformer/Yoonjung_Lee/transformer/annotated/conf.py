"""
conf.py
- 전체 학습 설정을 한 곳에 모아둔 파일입니다.
- 모델 크기, 배치 크기, epoch 수, 데이터 언어 방향 등을 여기서 바꿉니다.
"""

import torch


# 학습 장치 설정입니다.
# CUDA가 가능하면 GPU를 쓰고, 아니면 CPU를 사용합니다.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 저장소에서 기본으로 사용하는 배치 크기입니다.
# 한 번의 optimizer step에서 몇 개의 문장을 같이 처리할지 결정합니다.
batch_size = 128

# 최대 문장 길이입니다.
# positional encoding table을 이 길이만큼 미리 만들어 둡니다.
max_len = 256

# 모델 차원입니다.
# 논문에서 d_model로 부르는 값이며, attention과 FFN의 기본 벡터 크기입니다.
d_model = 512

# FFN 내부 은닉층 차원입니다.
# Transformer block 안의 position-wise feed-forward network가 512 → 2048 → 512 구조가 됩니다.
ffn_hidden = 2048

# multi-head attention의 head 개수입니다.
# d_model=512, n_head=8이면 head 하나당 차원은 64가 됩니다.
n_head = 8

# EncoderLayer와 DecoderLayer를 몇 층 쌓을지 결정합니다.
n_layers = 6

# dropout 확률입니다.
# 과적합을 줄이기 위해 attention/FFN/embedding 뒤에 적용됩니다.
drop_prob = 0.1

# 학습 반복 횟수입니다.
epochs = 1000

# 초기 learning rate입니다.
init_lr = 1e-5

# optimizer의 Adam beta 값입니다.
# Transformer 논문에서는 Adam + warmup schedule을 쓰지만, 이 코드는 일반 Adam 설정을 사용합니다.
adam_eps = 5e-9

# weight decay입니다.
# L2 정규화 효과를 줍니다.
weight_decay = 5e-4

# 학습 중간중간 loss를 출력할 간격입니다.
clip = 1

# torchtext Multi30k에서 사용할 파일 확장자입니다.
# ('.de', '.en')이면 독일어 source → 영어 target 번역입니다.
ext = ('.de', '.en')

# 문장 시작/끝 토큰입니다.
# torchtext Field가 각 문장 앞뒤에 자동으로 붙입니다.
init_token = '<sos>'
eos_token = '<eos>'

# vocab을 만들 때 최소 등장 횟수입니다.
# 2회 미만 등장한 단어는 <unk>로 처리됩니다.
min_freq = 2

# 모델 저장 경로입니다.
# 학습 중 validation loss가 가장 낮아지면 여기에 저장합니다.
model_path = 'saved/model-{0}.pt'
