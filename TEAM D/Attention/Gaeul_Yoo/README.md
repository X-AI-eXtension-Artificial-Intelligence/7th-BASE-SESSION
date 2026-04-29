

# 📝 NMT with Bahdanau Attention (French to English)

PyTorch를 사용하여 **Bahdanau Attention(Additive Attention)** 기반의 신경망 기계 번역(NMT) 모델을 구현하고 최적화한 프로젝트입니다. 프랑스어 문장을 영어로 번역하는 작업을 수행합니다.



## 🚀 Key Features & Improvements

단순한 Seq2Seq 모델을 넘어, 다음과 같은 성능 최적화 기법을 직접 적용하였습니다:

* **Attention Mechanism**: Bahdanau Attention을 구현하여 디코더가 인코더의 특정 시점에 집중할 수 있게 함.
* **Overfitting Prevention**: Encoder 및 Decoder에 **Dropout(p=0.1)** 레이어를 추가하여 모델의 일반화 성능 향상.
* **Learning Rate Optimization**: **ExponentialLR Scheduler**를 도입하여 학습 후반부의 손실 함수($Loss$)를 정교하게 수렴시킴.
* **Efficient Data Loading**: `TensorDataset`과 `RandomSampler`를 사용하여 배치 학습 효율을 높임.

## 📊 Training Results

20 Epoch 학습 결과, 매우 안정적인 수렴 곡선을 보여주었습니다.

* **Initial Loss (Epoch 1)**: 1.9534
* **Final Loss (Epoch 20)**: **0.0356** * **Final Learning Rate**: 0.000358



### Translation Examples
| Input (French) | Target (English) | Model Output |
| :--- | :--- | :--- |
| *je suis content que ca vous rende heureux .* | *i m glad that makes you happy .* | **i m glad that makes you happy .** |
| *tu es bon cuisinier non ?* | *you are a good cook aren t you ?* | **you are a good cook aren t you ?** |
| *je suis malade .* | *i m ill .* | **i am sick .** (Semantic Success) |

## 🛠 Tech Stack

* **Language**: Python 3.10
* **Framework**: PyTorch
* **Library**: NumPy, Matplotlib (for visualization), tqdm
* **Environment**: Miniconda (macOS / M1 Air)

## 📁 Project Structure

```text
Attention/
├── data/              # French-English dataset (fra.txt)
├── model.py           # Encoder, Bahdanau Attention, AttnDecoder
├── load_data.py       # Data preprocessing & Lang class
├── train.py           # Training loop with LR Scheduler
└── README.md          # Project documentation
```

## ⚙️ How to Run

1.  가상환경 활성화:
    ```bash
    conda activate attention_env
    ```
2.  필수 패키지 설치:
    ```bash
    pip install torch numpy matplotlib
    ```
3.  학습 실행:
    ```bash
    python train.py
    ```

젝트가 완벽하게 마무리됩니다! 다른 추가하고 싶은 내용이 있나요?
