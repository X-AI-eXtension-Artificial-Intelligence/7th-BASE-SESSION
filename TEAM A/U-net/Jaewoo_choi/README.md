U-Net 코드 분석 및 구현 정리
Based on hanyoseob/youtube-cnn-002-pytorch-unet
1. 개요

본 자료는 U-Net 논문의 핵심 아이디어를 실제 PyTorch 코드와 연결해서 설명하기 위한 정리본이다.
분석 대상 저장소는 model.py, dataset.py, train.py, util.py, data_read.py를 중심으로 구성되어 있으며, 전체 흐름은 데이터 전처리 → 데이터셋 구성 → U-Net 모델 정의 → 학습/검증 → 테스트 및 결과 저장 구조로 이루어져 있다. 저장소 루트에는 실제로 datasets, data_read.py, dataset.py, model.py, train.py, util.py, eval.py, display_results.py, run_unet.ipynb 등이 포함되어 있다.

U-Net은 원래 의료영상 분할(segmentation) 문제를 위해 제안된 구조로, 입력 영상의 각 픽셀에 대해 클래스 여부를 예측하는 dense prediction 모델이다. 이 저장소 역시 출력 채널을 1로 두고, 각 픽셀에 대해 binary segmentation을 수행하는 형태로 구현되어 있다. 마지막 레이어가 1x1 convolution으로 구성되고, 학습 손실로 BCEWithLogitsLoss를 사용하는 점이 이를 잘 보여준다.

2. U-Net 논문의 핵심 아이디어와 이 코드의 대응 관계

U-Net 논문의 핵심은 크게 세 가지로 볼 수 있다.

Encoder(Contracting Path): 해상도를 줄이면서 문맥 정보를 추출
Decoder(Expansive Path): 해상도를 복원하면서 픽셀 단위 위치 정보를 회복
Skip Connection: 같은 해상도의 encoder feature를 decoder에 직접 연결하여 localization 성능을 높임

이 저장소의 UNet 클래스는 이 구조를 거의 정석적으로 구현한다. enc1~enc5 계열이 encoder 역할을 하고, unpool과 dec 계열이 decoder 역할을 하며, forward 단계에서 torch.cat(...)으로 encoder feature와 decoder feature를 결합한다. 즉, 이 코드의 핵심은 단순 CNN이 아니라 **“압축하면서 의미를 배우고, 복원하면서 위치 정보를 되살리는 대칭형 구조”**라는 점이다.

3. 코드 전체 흐름

이 저장소의 실행 흐름은 다음과 같이 이해하면 된다.

data_read.py: 원본 TIFF 데이터를 읽어서 train, val, test용 .npy 파일로 분리
dataset.py: .npy 데이터를 불러오고 정규화/증강/텐서 변환 수행
model.py: U-Net 네트워크 정의
train.py: 학습, 검증, 테스트 루프 수행
util.py: 체크포인트 저장 및 불러오기 담당

즉, 논문 아이디어를 실제 실험 코드로 옮길 때 필요한 최소 요소들이 모듈별로 나뉘어 있는 구조다.

4. 핵심 코드 1: U-Net 기본 블록 정의

이 코드에서는 반복적으로 사용되는 합성곱 블록을 CBR2d로 구성한다. 구조는 Conv2d → BatchNorm2d → ReLU 순서다. 이 블록은 encoder와 decoder 대부분의 층에서 공통적으로 사용된다.

설명용 발췌
def CBR2d(in_channels, out_channels):
    return Sequential(
        Conv2d(...),
        BatchNorm2d(...),
        ReLU()
    )
해설

논문 관점에서 보면, 이 부분은 각 stage에서 feature map을 점점 더 풍부하게 만드는 기본 feature extractor 역할을 한다.
합성곱은 공간 패턴을 추출하고, Batch Normalization은 학습 안정성을 높이며, ReLU는 비선형성을 추가한다.
즉, U-Net의 성능은 단순히 skip connection 때문만이 아니라, 이런 convolution block들이 각 해상도 단계에서 충분한 표현력을 확보해주기 때문에 가능하다.

5. 핵심 코드 2: Encoder(Contracting Path)

이 저장소의 encoder는 채널 수를 64 → 128 → 256 → 512 → 1024로 늘려가며, 각 단계 사이에 MaxPool2d(kernel_size=2)를 사용해 해상도를 절반으로 줄인다. 이는 U-Net 논문에서 말하는 contracting path와 동일한 철학이다.

설명용 발췌
self.enc1_1, self.enc1_2
self.pool1
self.enc2_1, self.enc2_2
self.pool2
...
self.enc5_1
해설

여기서 중요한 것은 해상도는 줄이고 채널 수는 늘린다는 점이다.
왜냐하면 segmentation에서는 단순히 “어디에 경계가 있는가”만 아는 것이 아니라, 더 깊은 층에서 무엇이 의미 있는 구조인지도 알아야 하기 때문이다.
즉 encoder는 위치 해상도를 희생하는 대신 더 추상적이고 의미적인 feature를 학습한다. 이 과정 덕분에 네트워크는 단순 edge detector를 넘어서 객체 수준의 구조를 이해할 수 있게 된다.

6. 핵심 코드 3: Decoder(Expansive Path)

decoder에서는 ConvTranspose2d를 이용해 feature map을 다시 키운다. 이 저장소에서는 unpool4, unpool3, unpool2, unpool1이 그 역할을 담당한다. 이후 decoder block을 거치며 출력을 점차 segmentation mask에 가까운 형태로 복원한다.

설명용 발췌
self.unpool4 = nn.ConvTranspose2d(...)
self.dec4_2 = CBR2d(...)
self.dec4_1 = CBR2d(...)
해설

논문 관점에서 decoder는 단순 업샘플링이 아니다.
encoder에서 추출한 고수준 semantic feature를 이용해, 줄어들었던 해상도를 다시 키우면서 픽셀 단위 예측으로 되돌리는 과정이다.
즉 encoder가 “무엇인가”를 이해하는 쪽이라면, decoder는 “그것이 정확히 어디인가”를 복원하는 쪽이다. U-Net은 이 decoder 덕분에 classification이 아니라 segmentation에 특화된다.

7. 핵심 코드 4: Skip Connection

U-Net의 가장 상징적인 부분은 skip connection이다. 이 저장소에서도 decoder 업샘플 결과와 encoder feature를 torch.cat(..., dim=1)으로 결합한다. cat4, cat3, cat2, cat1이 바로 그 부분이다.

설명용 발췌
cat4 = torch.cat((unpool4, enc4_2), dim=1)
cat3 = torch.cat((unpool3, enc3_2), dim=1)
cat2 = torch.cat((unpool2, enc2_2), dim=1)
cat1 = torch.cat((unpool1, enc1_2), dim=1)
해설

이 부분이 중요한 이유는 encoder 깊은 층으로 갈수록 의미 정보는 풍부해지지만 위치 정보는 흐려지기 때문이다.
skip connection은 얕은 층의 고해상도 feature를 decoder에 직접 전달해서, 경계와 위치 같은 세밀한 정보를 보존하게 만든다.
즉, U-Net이 segmentation에서 강력한 이유는 단순히 깊은 네트워크라서가 아니라, semantic information과 localization information을 동시에 유지하는 구조이기 때문이다.

8. 핵심 코드 5: 최종 출력층

이 저장소의 마지막 출력은 nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)로 정의된다. 이는 각 픽셀 위치마다 1개의 logit 값을 출력해 binary mask를 예측한다는 의미다.

설명용 발췌
self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
해설

1x1 convolution은 공간 크기를 바꾸지 않고, 각 위치의 채널 정보를 결합해 최종 클래스를 예측하는 역할을 한다.
즉, 앞단에서 만든 다채로운 feature를 바탕으로 각 픽셀이 foreground인지 background인지를 판별하는 최종 분류기라고 볼 수 있다.
U-Net 논문 관점에서 보면 이 단계는 “dense feature map”을 “dense prediction mask”로 바꾸는 마지막 단계다.

9. 핵심 코드 6: Dataset 구성과 전처리

dataset.py에서는 파일명 기준으로 label*.npy, input*.npy를 분리해서 불러온다. 이후 label과 input을 255.0으로 나누어 정규화하고, 필요하면 채널 축을 추가한다. transform 단계에서는 Normalization, RandomFlip, ToTensor가 순차적으로 적용된다.

설명용 발췌
label = np.load(...)
input = np.load(...)
label = label / 255.0
input = input / 255.0
해설

이 부분은 모델 구조만큼 중요하다.
U-Net은 segmentation 모델이므로 입력 이미지뿐 아니라 정답 마스크도 픽셀 단위로 잘 정렬되어 있어야 한다.
또한 입력의 스케일을 맞춰야 학습이 안정적이므로, 이 코드에서는 먼저 0~1 범위로 스케일링한 뒤, 입력 영상에 한해서 평균 0.5, 표준편차 0.5 기준 정규화를 적용한다.
즉, 논문 구현에서 모델 구조가 “뼈대”라면, 데이터 전처리는 실제 성능을 좌우하는 “기초 공사”라고 볼 수 있다.

10. 핵심 코드 7: Data Augmentation

RandomFlip transform은 좌우 반전과 상하 반전을 랜덤하게 적용한다. segmentation에서는 입력과 정답 마스크가 정확히 같은 방식으로 변형되어야 하므로, 이 코드도 label과 input에 동일한 flip을 적용한다.

설명용 발췌
if np.random.rand() > 0.5:
    label = np.fliplr(label)
    input = np.fliplr(input)
해설

classification에서는 이미지에만 augmentation을 적용하면 되지만, segmentation에서는 입력과 마스크를 항상 함께 변환해야 한다.
이 부분은 segmentation 구현에서 자주 놓치기 쉬운 포인트이며, 잘못 구현하면 입력-정답 정렬이 깨져 학습이 망가진다.
따라서 이 transform 코드는 단순한 부가 기능이 아니라, segmentation 파이프라인의 정합성을 보장하는 핵심 요소다.

11. 핵심 코드 8: 손실함수와 학습 목표

train.py에서는 손실함수로 nn.BCEWithLogitsLoss()를 사용한다. 이는 최종 출력이 확률이 아닌 logit일 때 사용하는 binary classification용 손실이며, segmentation의 각 픽셀을 독립적인 이진 분류 문제처럼 다룬다.

설명용 발췌
fn_loss = nn.BCEWithLogitsLoss().to(device)
해설

이 저장소의 segmentation 문제는 다중 클래스가 아니라 binary mask 예측이므로, cross entropy의 다중 클래스 버전보다 BCE 계열이 더 자연스럽다.
또한 BCEWithLogitsLoss는 sigmoid와 BCE를 수치적으로 안정적으로 결합한 형태이므로, 마지막 층에서 별도로 sigmoid를 두지 않고도 학습할 수 있다.
즉, 이 코드는 “출력 1채널 + BCEWithLogitsLoss” 조합을 통해 U-Net을 binary segmentation 문제에 맞게 구현하고 있다.

12. 핵심 코드 9: 학습 루프

학습 루프는 매우 전형적인 PyTorch 구조를 따른다.
배치마다 input, label을 가져와 output = net(input)을 계산하고, loss를 구한 뒤, zero_grad → backward → step 순서로 파라미터를 업데이트한다. 검증 단계에서는 torch.no_grad()와 net.eval()을 사용해 gradient 계산 없이 성능을 확인한다.

설명용 발췌
output = net(input)
loss = fn_loss(output, label)
loss.backward()
optim.step()
해설

이 부분은 딥러닝 코드에서 가장 표준적인 학습 루프지만, U-Net 관점에서는 “이미지 전체를 넣고 픽셀 전체에 대한 loss를 한 번에 계산한다”는 점이 중요하다.
즉, 분류 모델처럼 한 장당 하나의 label을 맞추는 것이 아니라, 한 장의 이미지에 대해 수많은 픽셀 예측을 동시에 학습하는 구조다.
그래서 segmentation에서는 출력 크기와 label 크기가 정확히 일치해야 하고, 데이터셋과 transform 구현이 특히 중요해진다.

13. 핵심 코드 10: 테스트 및 결과 저장

테스트 모드에서는 가장 최근 체크포인트를 불러오고, 예측 결과를 png와 numpy 형식으로 저장한다. 이는 정량적 분석뿐 아니라 시각적 비교를 가능하게 해주는 실용적인 구성이다. 체크포인트 저장과 로드는 util.py의 save, load 함수가 담당한다.

설명용 발췌
save(...)
load(...)
plt.imsave(...)
np.save(...)
해설

segmentation에서는 숫자 loss만 보는 것보다, 실제 mask가 얼마나 자연스럽게 복원되었는지 직접 보는 것이 중요하다.
따라서 이 저장소는 테스트 결과를 이미지와 배열 둘 다 저장하도록 구성되어 있으며, 이는 논문 실험에서도 qualitative result와 quantitative result를 함께 제시하는 방식과 잘 맞는다.

14. 이 구현을 논문 관점에서 해석하면

이 저장소는 복잡한 변형 U-Net이 아니라, 기본형 U-Net의 핵심 아이디어를 교육용으로 비교적 직관적으로 보여주는 구현이라고 볼 수 있다.
구조적으로는 encoder-decoder 대칭 구조, skip connection, 1채널 binary mask 출력, BCE 기반 학습, 데이터 전처리 및 augmentation, 체크포인트 저장/복원 등 U-Net 실험에 필요한 핵심 요소가 모두 포함되어 있다.

즉, 이 코드는 “U-Net 논문을 실제 코드로 어떻게 옮기는가?”를 설명하기에 적절한 예시다.
특히 발표에서는 아래 메시지를 중심으로 정리하면 좋다.

Encoder는 문맥 정보를 압축적으로 추출한다.
Decoder는 해상도를 복원하며 픽셀 단위 예측을 수행한다.
Skip connection은 위치 정보를 보존해 segmentation 품질을 높인다.
데이터 전처리와 입력-마스크 정합성 유지가 segmentation 성능의 핵심이다.
최종적으로 이 구현은 binary segmentation용 기본 U-Net의 교육용 재현 예제라고 볼 수 있다.
