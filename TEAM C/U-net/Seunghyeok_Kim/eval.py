## 라이브러리 추가하기
import os                          # 파일/폴더 경로 처리용
import numpy as np                 # 배열 및 수치 연산용

import torch                       # PyTorch 메인 라이브러리
import torch.nn as nn              # 신경망 레이어 및 손실함수 모듈
from torch.utils.data import DataLoader   # 데이터 배치 로딩용
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 기록용 (현재 코드에서는 직접 사용되진 않음)

import matplotlib.pyplot as plt    # 이미지 저장 및 시각화용

from torchvision import transforms, datasets  # transform 조합용 / datasets (현재 코드에서는 직접 사용 안 함)


## 트레이닝 파라메터 설정하기
lr = 1e-3              # learning rate
batch_size = 4         # 한 번에 불러올 데이터 개수
num_epoch = 100        # 전체 epoch 수 (현재 test 코드에서는 직접 사용 안 함)

# data_dir = './datasets'
# ckpt_dir = './checkpoint'
# log_dir = './log'
# result_dir = './results'
# 로컬 경로 예시. 현재는 아래 Google Drive 경로를 사용함

data_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/datasets'      # 데이터셋 경로
ckpt_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/checkpoint'    # 모델 체크포인트 저장 경로
log_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/log'            # 로그 저장 경로
result_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/results'     # 결과 이미지/배열 저장 경로

# 결과 저장 폴더가 없으면 새로 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))      # PNG 결과 저장 폴더 생성
    os.makedirs(os.path.join(result_dir, 'numpy'))    # NPY 결과 저장 폴더 생성

# CUDA 사용 가능하면 GPU, 아니면 CPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()   # nn.Module 초기화

        # Conv + BatchNorm + ReLU 블록을 만드는 함수
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []   # 레이어들을 담을 리스트

            # 2D convolution layer 추가
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]

            # 배치 정규화 추가
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            # 활성화 함수 ReLU 추가
            layers += [nn.ReLU()]

            # 위 레이어들을 순차적으로 실행하는 Sequential로 묶음
            cbr = nn.Sequential(*layers)

            return cbr   # Conv-BN-ReLU 블록 반환

        # -----------------------------
        # Contracting path (Encoder)
        # -----------------------------

        # 입력 채널 1 -> 64채널
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)

        # 64채널 -> 64채널
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # 해상도 절반으로 줄이는 max pooling
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 64채널 -> 128채널
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)

        # 128채널 -> 128채널
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        # 해상도 절반으로 줄임
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 128채널 -> 256채널
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)

        # 256채널 -> 256채널
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        # 해상도 절반으로 줄임
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 256채널 -> 512채널
        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)

        # 512채널 -> 512채널
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        # 해상도 절반으로 줄임
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # bottleneck 부분: 512채널 -> 1024채널
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # -----------------------------
        # Expansive path (Decoder)
        # -----------------------------

        # bottleneck feature를 decoder 시작 채널 수에 맞게 1024 -> 512로 줄임
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        # transpose convolution으로 해상도를 2배로 복원
        # 채널 수는 512 -> 512 유지
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # skip connection으로 encoder의 enc4_2(512채널)와 concat하면
        # 총 1024채널이 되므로 입력 채널은 2 * 512
        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)

        # 512채널 -> 256채널로 줄여 다음 decoder 단계로 넘김
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        # 해상도 2배 복원, 채널 수 256 유지
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # concat 후 256 + 256 = 512채널 입력
        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)

        # 256채널 -> 128채널
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        # 해상도 2배 복원, 채널 수 128 유지
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # concat 후 128 + 128 = 256채널 입력
        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)

        # 128채널 -> 64채널
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        # 해상도 2배 복원, 채널 수 64 유지
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        # concat 후 64 + 64 = 128채널 입력
        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)

        # 64채널 -> 64채널
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        # 마지막 1x1 convolution
        # 최종 출력 채널을 1개로 만들어 segmentation map 생성
        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        # -----------------------------
        # Encoder
        # -----------------------------

        enc1_1 = self.enc1_1(x)        # 입력 -> 첫 번째 conv block
        enc1_2 = self.enc1_2(enc1_1)   # 두 번째 conv block
        pool1 = self.pool1(enc1_2)     # 해상도 절반 축소

        enc2_1 = self.enc2_1(pool1)    # 64 -> 128
        enc2_2 = self.enc2_2(enc2_1)   # 128 유지
        pool2 = self.pool2(enc2_2)     # 해상도 절반 축소

        enc3_1 = self.enc3_1(pool2)    # 128 -> 256
        enc3_2 = self.enc3_2(enc3_1)   # 256 유지
        pool3 = self.pool3(enc3_2)     # 해상도 절반 축소

        enc4_1 = self.enc4_1(pool3)    # 256 -> 512
        enc4_2 = self.enc4_2(enc4_1)   # 512 유지
        pool4 = self.pool4(enc4_2)     # 해상도 절반 축소

        enc5_1 = self.enc5_1(pool4)    # bottleneck: 512 -> 1024

        # -----------------------------
        # Decoder
        # -----------------------------

        dec5_1 = self.dec5_1(enc5_1)   # 1024 -> 512로 channel 축소

        # 업샘플링 후 encoder feature와 concat
        unpool4 = self.unpool4(dec5_1)             # 해상도 2배 복원
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # 채널 기준(concat dim=1)으로 skip connection 결합
        dec4_2 = self.dec4_2(cat4)                 # 1024 -> 512
        dec4_1 = self.dec4_1(dec4_2)               # 512 -> 256

        # 다음 decoder 단계
        unpool3 = self.unpool3(dec4_1)             # 해상도 2배 복원
        cat3 = torch.cat((unpool3, enc3_2), dim=1) # 256 + 256 = 512채널
        dec3_2 = self.dec3_2(cat3)                 # 512 -> 256
        dec3_1 = self.dec3_1(dec3_2)               # 256 -> 128

        # 다음 decoder 단계
        unpool2 = self.unpool2(dec3_1)             # 해상도 2배 복원
        cat2 = torch.cat((unpool2, enc2_2), dim=1) # 128 + 128 = 256채널
        dec2_2 = self.dec2_2(cat2)                 # 256 -> 128
        dec2_1 = self.dec2_1(dec2_2)               # 128 -> 64

        # 마지막 decoder 단계
        unpool1 = self.unpool1(dec2_1)             # 해상도 2배 복원
        cat1 = torch.cat((unpool1, enc1_2), dim=1) # 64 + 64 = 128채널
        dec1_2 = self.dec1_2(cat1)                 # 128 -> 64
        dec1_1 = self.dec1_1(dec1_2)               # 64 -> 64

        # 최종 segmentation output 생성
        x = self.fc(dec1_1)                        # 64 -> 1

        return x                                   # logits 반환


## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir       # 데이터 경로 저장
        self.transform = transform     # transform 저장

        lst_data = os.listdir(self.data_dir)   # 폴더 내 전체 파일 목록 읽기

        # 파일명 prefix가 label인 파일만 추출
        lst_label = [f for f in lst_data if f.startswith('label')]

        # 파일명 prefix가 input인 파일만 추출
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 정렬해서 input-label 순서 맞추기
        lst_label.sort()
        lst_input.sort()

        # 클래스 변수로 저장
        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)   # 전체 데이터 개수 반환

    def __getitem__(self, index):
        # index에 해당하는 label 파일 로드
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        # index에 해당하는 input 파일 로드
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 0~255 범위를 0~1 범위로 정규화
        label = label / 255.0
        input = input / 255.0

        # 만약 2차원 배열(H, W)이면 channel 축을 추가해서 (H, W, 1)로 변경
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # input과 label을 딕셔너리 형태로 묶음
        data = {'input': input, 'label': label}

        # transform이 정의되어 있으면 transform 적용
        if self.transform:
            data = self.transform(data)

        return data   # 하나의 샘플 반환


## 트렌스폼 구현하기
class ToTensor(object):
    def __call__(self, data):
        # 딕셔너리에서 label과 input 추출
        label, input = data['label'], data['input']

        # (H, W, C) -> (C, H, W) 형태로 변경 후 float32로 변환
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # numpy 배열을 torch tensor로 변환
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean   # 평균 저장
        self.std = std     # 표준편차 저장

    def __call__(self, data):
        # label과 input 추출
        label, input = data['label'], data['input']

        # 입력 영상만 정규화
        # (x - mean) / std
        input = (input - self.mean) / self.std

        # label은 그대로 두고 input만 정규화된 값으로 저장
        data = {'label': label, 'input': input}

        return data


class RandomFlip(object):
    def __call__(self, data):
        # label과 input 추출
        label, input = data['label'], data['input']

        # 50% 확률로 좌우 반전
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        # 50% 확률로 상하 반전
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        # 반전된 데이터 다시 저장
        data = {'label': label, 'input': input}

        return data


## 네트워크 학습하기
# 입력 데이터에 대해 정규화 후 tensor 변환 수행
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

# test 폴더의 데이터를 Dataset 객체로 생성
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)

# DataLoader 생성
# shuffle=False 이므로 test 데이터 순서를 유지
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)


## 네트워크 생성하기
net = UNet().to(device)   # U-Net 모델 생성 후 device(GPU/CPU)로 이동


## 손실함수 정의하기
# BCEWithLogitsLoss:
# sigmoid + binary cross entropy를 한 번에 처리하는 손실함수
fn_loss = nn.BCEWithLogitsLoss().to(device)


## Optimizer 설정하기
# Adam optimizer 사용
optim = torch.optim.Adam(net.parameters(), lr=lr)


## 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)   # 전체 test 데이터 개수

# 전체 test batch 수 계산
num_batch_test = np.ceil(num_data_test / batch_size)


## 그밖에 부수적인 functions 설정하기
# tensor -> numpy 변환 후 (B, C, H, W) -> (B, H, W, C)로 변경
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

# 정규화된 이미지를 다시 원래 스케일로 복원
fn_denorm = lambda x, mean, std: (x * std) + mean

# 0.5 기준으로 binary classification
fn_class = lambda x: 1.0 * (x > 0.5)


## 네트워크 저장하기
def save(ckpt_dir, net, optim, epoch):
    # 체크포인트 폴더가 없으면 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # 모델 가중치와 optimizer 상태를 함께 저장
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    # 체크포인트 폴더가 없으면 처음부터 시작
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    # 체크포인트 파일 목록 불러오기
    ckpt_lst = os.listdir(ckpt_dir)

    # 파일명 안 숫자를 기준으로 정렬
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    # 가장 마지막(최신) 체크포인트 불러오기
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    # 모델 가중치 복원
    net.load_state_dict(dict_model['net'])

    # optimizer 상태 복원
    optim.load_state_dict(dict_model['optim'])

    # 파일명에서 epoch 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch   # 복원된 모델, optimizer, epoch 반환


## 네트워크 학습시키기
st_epoch = 0   # 시작 epoch 초기값

# 가장 최근 체크포인트가 있으면 불러오기
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# gradient 계산 비활성화: test/inference 단계
with torch.no_grad():
    net.eval()       # evaluation mode로 전환
    loss_arr = []    # batch별 loss 저장용 리스트

    # test dataloader 순회
    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)   # label을 device로 이동
        input = data['input'].to(device)   # input을 device로 이동

        output = net(input)   # 모델 출력 생성

        # 손실함수 계산하기
        loss = fn_loss(output, label)   # output과 label 사이 BCE loss 계산

        loss_arr += [loss.item()]       # 현재 batch loss 저장

        # 현재까지 평균 loss 출력
        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        # 결과 저장을 위해 numpy로 변환
        label = fn_tonumpy(label)                                 # label tensor -> numpy
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))  # input 역정규화 후 numpy 변환
        output = fn_tonumpy(fn_class(output))                    # output 이진화 후 numpy 변환

        # batch 안 각 샘플마다 저장
        for j in range(label.shape[0]):
            # 전체 데이터 기준 저장 id 계산
            id = num_batch_test * (batch - 1) + j

            # PNG 이미지 저장
            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            # NPY 배열 저장
            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

# 전체 test set 평균 loss 출력
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))
