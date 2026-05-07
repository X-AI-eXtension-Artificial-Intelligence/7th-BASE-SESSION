
# 📑 Transformer 기반 불어-영어 기계 번역 프로젝트

본 프로젝트는 Transformer 아키텍처를 직접 구현하고, Multi30k 데이터셋(불어-영어)을 활용하여 신경망 기계 번역(NMT) 모델을 구축한 과정을 담고 있습니다.

## 🛠 Tech Stack
* **Framework**: PyTorch
* **Environment**: Google Colab (CUDA 지원)
* **Libraries**: torchtext, math, tqdm

## 🏗 Model Architecture
표준 Transformer 구조를 따르며, 학습 효율을 위해 3개의 인코더/디코더 레이어를 사용했습니다.
* **Embedding Size**: 512
* **Attention Heads**: 8
* **Forward Expansion**: 2 (1024 dims)
* **Optimization**: Adam Optimizer, Label Smoothing 적용



---

## 🔍 Trouble Shooting (핵심 기록 사항)

학습 완료 후 추론(Inference) 단계에서 발생한 **Model State Dict Mismatch** 문제를 해결하며 모델의 내부 구조와 가중치 로딩 메커니즘을 심도 있게 이해할 수 있었습니다.

### 1. Key Mismatch: "queries_linear" vs "query"
* **문제**: `model.pth` 파일에 저장된 가중치 이름은 `queries_linear`였으나, 테스트 스크립트의 클래스 정의에서는 `query`라는 이름을 기대하여 `RuntimeError` 발생.
* **해결**: 학습 코드의 명명 규칙(Naming Convention)을 분석하여 `model.py` 내의 Linear Layer 변수명을 모델 파일과 동일하게 동기화.

### 2. 학습 코드 내 오타(Typo) 대응
* **문제**: 학습 단계에서 `feed_forawrd`라는 오타가 포함된 상태로 가중치가 저장됨. 이를 `feed_forward`로 수정 시 가중치 로딩 불가.
* **해결**: 이미 학습된 모델의 가중치 이름표(Key)를 유지하기 위해 의도적으로 오타를 유지하거나, 가중치 딕셔너리의 키값을 매핑하여 해결. **(데이터 일관성의 중요성 체득)**

### 3. Inference 메서드(encode/decode) 분리
* **문제**: `Transformer` 클래스 내에 `forward`만 정의되어 있어, 한 단어씩 생성해야 하는 테스트 단계의 `encode` 함수 호출 시 `AttributeError` 발생.
* **해결**: 인코더의 출력값을 고정한 채 디코더를 반복 실행할 수 있도록 `encode()`와 `decode()` 메서드를 별도로 구현하여 추론 로직 완성.

---

## 📈 Learning Results
* **Data Size**: 약 29,000개의 문장 쌍
* **Training Epochs**: 10 Epochs
* **Observation**: 제한된 데이터셋으로 인해 일부 문장에서 단어 반복(Repeat) 현상이 관찰됨. 향후 Beam Search 도입 및 학습 데이터 증설을 통해 성능 개선 예정.

---

## 🚀 How to Run
```bash
# 모델 추론 실행
python test.py
```

