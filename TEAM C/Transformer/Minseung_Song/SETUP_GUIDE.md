# Transformer 실행 가이드 (학회용 빠른 워크스루)

이 가이드는 `mytransformer/` 안의 코드를 처음부터 끝까지 한 번 돌려보기 위한 단계별 매뉴얼입니다. 순서대로 따라가면 됩니다.

> **사용자 환경 메모**: RTX 4060 Laptop GPU (8GB VRAM) + AMD Radeon 780M(내장).
> 학습은 RTX 4060(GPU 1)에서 진행되며, batch_size 128 / max_len 256 그대로 두고 돌려도 VRAM 여유 있음.
> RTX 40-series는 sm_89 아키텍처라 torch 1.10(cu113)에서 첫 실행 시 sm 호환 경고가 뜰 수 있는데, PTX JIT으로 정상 동작합니다(자세한 내용은 10번 문제 해결 참고).

---

## 0. 결론부터: 가능한가?

**가능합니다.** 단, 다음 조건이 필요합니다.

- Python **3.9** 권장 (3.8도 OK, 3.10 이상은 torch 1.10이 지원 안 함)
- 가상환경 사용 권장 (다른 프로젝트와 의존성 충돌 방지)
- **NVIDIA GPU + CUDA 11.x 드라이버** (없으면 CPU로도 돌아가지만 매우 느림)

이 코드는 2019년 작성되어 `torchtext.legacy` API를 쓰는데, 이 API는 torchtext 0.12부터 제거되었습니다. 그래서 옛날 버전(torch 1.10 + torchtext 0.11)을 핀해야 합니다. 그리고 Multi30k 원본 다운로드 URL이 죽어 있어서 GitHub 미러로 패치해뒀습니다(`util/data_loader.py`).

추가/수정한 파일:

- `requirements.txt` — 핀된 패키지 버전 목록 (torch는 별도 명령으로 설치)
- `util/data_loader.py` — Multi30k URL 미러로 교체
- `result/`, `saved/` — train.py가 결과물을 저장하는 폴더 (빈 폴더 생성)
- `test.py` — 학습 후 추론용 스크립트 (원본에 없었음)
- `check_env.py` — 환경(GPU 포함) 점검용 스크립트

---

## 1. Python 3.9 / GPU 드라이버 확인

VS Code 터미널에서:

```
py -0
```

`-V:3.9` 항목이 보이면 OK. 없으면 https://www.python.org/downloads/release/python-3913/ 에서 Windows installer 받아 설치 (Add to PATH 체크). 설치 후 다시 `py -0`로 확인.

GPU 드라이버 확인:

```
nvidia-smi
```

표가 출력되면서 상단에 `CUDA Version: 11.x` (또는 12.x)가 보이면 OK. `nvidia-smi`가 명령을 못 찾는다고 나오면 NVIDIA 드라이버부터 설치(https://www.nvidia.com/Download/index.aspx).

> CUDA Toolkit 자체는 따로 설치할 필요 없습니다. PyTorch wheel이 필요한 CUDA 런타임을 같이 담고 있어서, 드라이버만 최신이면 됩니다. `CUDA Version: 12.x`가 나와도 PyTorch 1.10 + cu113 wheel은 정상 작동합니다(상위 호환).

---

## 2. 가상환경 생성

`mytransformer/` 폴더에서 (VS Code 터미널 cwd가 거기인지 확인):

```
py -3.9 -m venv trans
```

`trans/` 폴더가 생성됩니다.

---

## 3. 가상환경 활성화

PowerShell:

```
trans\Scripts\Activate.ps1
```

cmd:

```
trans\Scripts\activate.bat
```

프롬프트 앞에 `(trans)`가 붙으면 성공. (PowerShell에서 실행 정책 오류가 나면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 한 번 실행 후 재시도)

VS Code가 자동으로 `trans`를 인식하지 못하면 `Ctrl+Shift+P` → "Python: Select Interpreter" → `trans\Scripts\python.exe` 선택.

---

## 4. 패키지 설치

가상환경 활성화된 상태에서:

```
python -m pip install --upgrade pip
```

먼저 **GPU용 torch + torchtext**를 따로 설치합니다 (CUDA 11.3 wheel, 대부분의 RTX/GTX GPU에서 동작):

```
pip install torch==1.10.0+cu113 torchtext==0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

> 이 명령이 실패하거나 너무 오래된 GPU(GTX 9xx 등)면 `+cu111` 버전을 시도하세요:
>
> ```
> pip install torch==1.10.0+cu111 torchtext==0.11.0 -f https://download.pytorch.org/whl/cu111/torch_stable.html
> ```
>
> GPU가 없거나 CUDA 설치가 어렵다면 CPU 빌드:
>
> ```
> pip install torch==1.10.0 torchtext==0.11.0
> ```

torch wheel은 2GB 가까이 되니 시간이 좀 걸립니다. 끝나면 나머지 패키지 설치:

```
pip install -r requirements.txt
```

이어서 spaCy 토크나이저 모델 다운로드 (영어/독일어, 각 15MB):

```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

마지막으로 환경 점검:

```
python check_env.py
```

출력 예시:

```
Python: 3.9.13
==================================================
torch         : 1.10.0+cu113
  CUDA build  : 11.3
  CUDA avail  : True
  Device count: 1
  [0] NVIDIA GeForce RTX 3060 Laptop GPU (compute 8.6, 6.0 GB)
torchtext     : 0.11.0
  legacy API  : OK
spacy         : 3.4.4
  en_core_web_sm    : OK
  de_core_news_sm   : OK
==================================================
모두 OK이면 train.py 실행 준비 완료.
```

`CUDA avail : True`가 나오는 게 핵심. `False`로 나오면 CPU 빌드가 깔린 것이니 위 GPU 명령으로 다시 설치하세요.

---

## 5. 빠른 테스트용 conf.py 수정

`conf.py`는 epoch=1000으로 되어 있습니다. 한 번 돌려보는 용도라면 줄여주세요.

`conf.py` 열어서:

```python
epoch = 1000      # → 10~30 정도로 변경 (학회 시연용이면 10이면 충분)
batch_size = 128  # RTX 4060(8GB)이면 그대로 OK
```

RTX 4060(8GB) 기준 예상 VRAM 사용량은 약 2~4GB(모델 220MB + 옵티마이저 상태 + 활성화), batch_size 128 / max_len 256으로 충분히 들어갑니다.

VRAM 별 batch_size 가이드 (참고):

| VRAM      | 권장 batch_size   |
| --------- | ----------------- |
| 4GB 이하  | 32~64             |
| 6~8GB     | 128 (그대로)      |
| 12GB 이상 | 192~256 까지 가능 |

OOM(out of memory) 에러가 나면 batch_size를 절반으로 줄이고 재시도.

---

## 6. 학습 실행

```
python train.py
```

처음 실행 시 Multi30k 데이터셋(약 2MB)을 자동으로 다운로드합니다(`.data/multi30k/` 폴더 생성). 만약 다운로드 단계에서 실패한다면 5번 단계 다시 확인 후, `util/data_loader.py`의 URL이 패치되어 있는지 점검.

정상 출력 예시:

```
dataset initializing start
dataset initializing done
The model has 55,207,087 trainable parameters
step : 0.0 % , loss : 9.234...
step : 0.79 % , loss : 8.871...
...
Epoch: 1 | Time: 3m 12s
    Train Loss: 6.823 | Train PPL: 920.5
    Val Loss: 5.912 |  Val PPL: 369.7
    BLEU Score: 0.012
```

epoch이 진행될수록 loss가 떨어지는 게 보이면 성공. 매 epoch 끝나면 valid loss가 갱신될 때마다 `saved/model-<loss>.pt` 파일이 저장됩니다.

---

## 7. 학습 중단/재개

학습 중간에 멈추고 싶으면 `Ctrl+C`. 이미 저장된 체크포인트(`saved/model-*.pt`)는 그대로 남아 있으니 8단계로 넘어가도 됩니다.

---

## 8. 추론 (테스트) 실행

학습이 끝났거나 중간에 멈췄으면, 저장된 체크포인트로 번역을 돌려봅니다.

```
python test.py
```

옵션:

```
python test.py --num_samples 10                    # 샘플 10개 출력
python test.py --checkpoint saved/model-3.20.pt    # 특정 체크포인트 지정
```

출력 예시:

```
[INFO] Loading checkpoint: saved/model-3.204.pt
[INFO] Device: cpu

===== Sample Translations (EN -> DE) =====

[1]
  SRC : a man in an orange hat starring at something
  TRG : ein mann mit einem orangen hut der etwas anstarrt
  PRED: ein mann mit einem orangefarbenen hut starrt auf etwas
...

[INFO] Computing BLEU on full test set...

===== Test BLEU: 12.345 =====
```

epoch 10번만 돌렸으면 BLEU가 5~15 정도, 100 epoch 이상 돌리면 20 이상. 원 저자의 1000 epoch 결과는 26.4.

---

## 9. (선택) 학습 곡선 그리기

```
python graph.py
```

`result/train_loss.txt`, `result/test_loss.txt`, `result/bleu.txt`를 읽어서 그래프 표시. matplotlib 창이 뜹니다.

---

## 10. 문제 해결

| 증상                                                                                      | 원인 / 해결                                                                                                                                               |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'torchtext.legacy'`                                 | torchtext 버전이 0.12 이상. `pip install torchtext==0.11.0`                                                                                               |
| `URLError: <urlopen error [Errno -2]>` 또는 404                                           | Multi30k URL 패치 누락. `util/data_loader.py`에 `Multi30k.urls = [...]` 블록이 있는지 확인                                                                |
| `OSError: [E050] Can't find model 'en_core_web_sm'`                                       | `python -m spacy download en_core_web_sm` 다시 실행                                                                                                       |
| `RuntimeError: Expected all tensors to be on the same device`                             | conf.py의 `device` 설정 확인. CUDA 안 쓰는데 잘못 잡혔으면 `device = torch.device("cpu")`로 강제                                                          |
| `check_env.py`에서 `CUDA avail : False`                                                   | CPU 빌드가 깔린 상태. `pip uninstall torch torchtext` 후 4단계 GPU 설치 명령 다시 실행                                                                    |
| `nvidia-smi: command not found`                                                           | NVIDIA 드라이버 미설치. https://www.nvidia.com/Download/index.aspx 에서 GPU에 맞는 드라이버 설치                                                          |
| `NVIDIA GeForce RTX 4060 Laptop GPU with CUDA capability sm_89 is not compatible...` 경고 | RTX 40-series + torch 1.10(cu113)의 알려진 경고. **학습은 정상 진행됨**(PTX JIT). 무시해도 됨. 첫 에폭이 살짝 느리고 전체 약 10~15% 성능 손실 정도가 전부 |
| `RuntimeError: CUDA error: no kernel image is available` (실제 에러)                      | PTX JIT마저 실패한 경우. 드물지만 가능. 이때는 코드를 모던 torch 2.x + 수동 데이터로더로 옮겨야 함(별도 요청)                                             |
| `CUDA out of memory`                                                                      | `conf.py`의 `batch_size`를 절반으로 줄이세요 (128→64→32→16). 그래도 안 되면 `max_len`을 256→128로                                                         |
| `MemoryError` (CPU)                                                                       | RAM 부족. `batch_size`를 32 이하로                                                                                                                        |
| `FileNotFoundError: result/train_loss.txt`                                                | `result/` 폴더가 비어 있음. 빈 폴더는 만들어 뒀으니 train.py 1 epoch만 돌면 자동 생성                                                                     |
| 학습이 너무 느림                                                                          | `epoch`을 10~30으로 줄이세요. 어차피 한 번 돌려보는 용도                                                                                                  |

---

## 11. 전체 한 줄 요약 (GPU 기준, PowerShell)

```
py -3.9 -m venv trans
trans\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==1.10.0+cu113 torchtext==0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python check_env.py
# conf.py 에서 epoch=10으로 수정
python train.py
python test.py
```

이 순서대로 막힘 없이 돌면 끝. 막히는 단계 있으면 알려주세요.
