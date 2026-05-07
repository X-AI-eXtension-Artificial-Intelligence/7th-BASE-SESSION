import torch

# 장치 설정 (GPU 사용 가능 여부에 따라 CUDA 또는 CPU 선택)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델 하이퍼파라미터 설정
batch_size = 128
max_len = 256
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# 학습 하이퍼파라미터 설정
init_lr = 1e-5
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
