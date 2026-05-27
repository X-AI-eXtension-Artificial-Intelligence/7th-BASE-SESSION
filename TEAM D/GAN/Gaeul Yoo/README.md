# GAN — Goodfellow et al. 2014

PyTorch implementation of **Generative Adversarial Nets** (Goodfellow et al., NeurIPS 2014).  
논문: https://arxiv.org/abs/1406.2661

---

## 결과

| Epoch 10 | Epoch 100 | Epoch 200 |
|---|---|---|
| `outputs/samples_epoch_0010.png` | `outputs/samples_epoch_0100.png` | `outputs/samples_epoch_0200.png` |

**최종 Loss (Epoch 200)**
- D_loss: 1.1704
- G_loss: 1.0467

---

## 아키텍처

논문 Section 3의 MLP 기반 구조를 그대로 구현했습니다.

### Generator
```
z ~ N(0,1) [100-dim]
→ Linear(100 → 256) + LeakyReLU(0.2)
→ Linear(256 → 512) + LeakyReLU(0.2)
→ Linear(512 → 1024) + LeakyReLU(0.2)
→ Linear(1024 → 784) + Tanh
→ output: 28×28 image, range [-1, 1]
```

### Discriminator
```
input: 28×28 flattened image [784-dim]
→ Linear(784 → 1024) + LeakyReLU(0.2) + Dropout(0.3)
→ Linear(1024 → 512) + LeakyReLU(0.2) + Dropout(0.3)
→ Linear(512 → 256)  + LeakyReLU(0.2) + Dropout(0.3)
→ Linear(256 → 1)    + Sigmoid
→ output: probability (real=1, fake=0)
```

---

## 학습 방법

논문 Algorithm 1, k=1 (D 1회 업데이트 per G 1회 업데이트)

```
D step: BCELoss(D(real), 1) + BCELoss(D(G(z).detach()), 0)
G step: BCELoss(D(G(z)), 1)   ← non-saturating variant
```

| 하이퍼파라미터 | 값 |
|---|---|
| Epochs | 200 |
| Batch size | 64 |
| Learning rate | 0.0002 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| Latent dim | 100 |
| Dataset | MNIST |

---

## 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습
python main.py

# 옵션 지정
python main.py --epochs 200 --batch_size 64 --lr 0.0002 --latent_dim 100
```

**출력 파일**

```
outputs/
├── samples_epoch_0010.png   # 10 epoch마다 생성 이미지 (fixed z)
├── samples_epoch_0020.png
├── ...
├── samples_epoch_0200.png
├── loss_curve.png           # D/G loss 곡선
├── G_final.pth              # 최종 Generator 가중치
├── D_final.pth              # 최종 Discriminator 가중치
└── checkpoints/
    ├── G_epoch_0050.pth
    ├── D_epoch_0050.pth
    └── ...
```

---

## 파일 구조

```
GAN/
├── models.py     # Generator, Discriminator 정의
├── train.py      # 학습 루프 (Algorithm 1)
├── utils.py      # Loss curve 시각화
├── main.py       # 진입점, 하이퍼파라미터
├── GAN_MNIST.ipynb  # Colab 실행용 노트북
├── requirements.txt
└── README.md
```

---

## 논문 참고

```bibtex
@inproceedings{goodfellow2014generative,
  title={Generative Adversarial Nets},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and
          Xu, Bing and Warde-Farley, David and Ozair, Sherjil and
          Courville, Aaron and Bengio, Yoshua},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2014}
}
```
