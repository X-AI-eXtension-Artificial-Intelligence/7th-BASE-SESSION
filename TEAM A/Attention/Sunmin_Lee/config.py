class Config:
    # 모델
    input_size   = 10        # 입력 숫자 범위 (0~9)
    output_size  = 10
    embed_dim    = 32
    hidden_size  = 64
    n_layers     = 1

    # 학습
    seq_len      = 7         # 입력 시퀀스 길이
    batch_size   = 64
    epochs       = 30
    lr           = 1e-3

    # 기타
    device       = "cpu"     # "cuda" 가능
    seed         = 42