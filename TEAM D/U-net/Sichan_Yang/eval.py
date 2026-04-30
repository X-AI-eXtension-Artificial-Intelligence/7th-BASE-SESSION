import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from torchvision import transforms

## 하이퍼파라미터 설정
lr = 1e-3  # 학습률
batch_size = 4  # 배치 크기
num_epoch = 100  # 총 에폭 수

# 경로 설정
data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

# 결과 저장 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))  # 시각화 결과
    os.makedirs(os.path.join(result_dir, 'numpy'))  # numpy 결과

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## UNet 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Conv-BatchNorm-ReLU 블록 정의
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias),
                      nn.BatchNorm2d(out_channels),
                      nn.ReLU()]
            return nn.Sequential(*layers)

        # 인코딩 경로
        self.enc1_1 = CBR2d(1, 64)
        self.enc1_2 = CBR2d(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = CBR2d(64, 128)
        self.enc2_2 = CBR2d(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = CBR2d(128, 256)
        self.enc3_2 = CBR2d(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = CBR2d(256, 512)
        self.enc4_2 = CBR2d(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = CBR2d(512, 1024)  # bottleneck

        # 디코딩 경로
        self.dec5_1 = CBR2d(1024, 512)
        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, 2)

        self.dec4_2 = CBR2d(1024, 512)
        self.dec4_1 = CBR2d(512, 256)
        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, 2)

        self.dec3_2 = CBR2d(512, 256)
        self.dec3_1 = CBR2d(256, 128)
        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, 2)

        self.dec2_2 = CBR2d(256, 128)
        self.dec2_1 = CBR2d(128, 64)
        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, 2)

        self.dec1_2 = CBR2d(128, 64)
        self.dec1_1 = CBR2d(64, 64)

        self.fc = nn.Conv2d(64, 1, 1)  # 출력 채널 1 (binary segmentation)

    def forward(self, x):
        # 인코딩
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # 디코딩
        dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)  # 최종 출력

        return x

## 커스텀 데이터셋 클래스
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        lst_data = os.listdir(self.data_dir)
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index])) / 255.0
        input = np.load(os.path.join(self.data_dir, self.lst_input[index])) / 255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}
        if self.transform:
            data = self.transform(data)

        return data

## 텐서 변환 클래스
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

## 정규화 클래스
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        return {'label': label, 'input': input}

## 랜덤 좌우/상하 반전 클래스
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        return {'label': label, 'input': input}

## 전처리 정의
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

# 테스트 데이터셋 로딩
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

## 네트워크 생성 및 손실 함수, 옵티마이저 설정
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 부수 함수 및 변수
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## 모델 저장 함수
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 모델 불러오기 함수
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        return net, optim, 0
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

## 테스트 수행
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = net(input)
        loss = fn_loss(output, label)
        loss_arr += [loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = int(num_batch_test * (batch - 1) + j)
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output[j].squeeze(), cmap='gray')
            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output[j].squeeze())

# 평균 손실 출력
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))
