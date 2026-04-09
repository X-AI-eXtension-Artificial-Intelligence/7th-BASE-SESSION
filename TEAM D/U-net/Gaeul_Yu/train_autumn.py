## 라이브러리 추가하기
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

## 1. 네트워크 구축하기
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                          kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU()
            )

        # Contracting Path       
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256) 
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expanding Path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=512, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True) 

        self.dec2_2 = CBR2d(in_channels=256, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)   

        self.dec1_2 = CBR2d(in_channels=128, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, x):
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

        x = self.fc(dec1_1)
        return x

## 2. 데이터셋 및 트랜스폼 정의
class AutumnDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        lst_data = os.listdir(self.data_dir)
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, idx):
        label = np.load(os.path.join(self.data_dir, self.lst_label[idx]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[idx]))
        label, input = label/255.0, input/255.0

        if label.ndim == 2: label = label[:, :, np.newaxis]
        if input.ndim == 2: input = input[:, :, np.newaxis]

        data = {'label': label, 'input': input}
        if self.transform: data = self.transform(data)
        return data

class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean, self.std = mean, std
    def __call__(self, data):
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std
        return {'label': label, 'input': input}
    
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        if np.random.rand() > 0.5:
            label, input = np.fliplr(label), np.fliplr(input)
        if np.random.rand() > 0.5:
            label, input = np.flipud(label), np.flipud(input)
        return {'label': label.copy(), 'input': input.copy()}

## 3. 부수적인 함수들
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, 
               os.path.join(ckpt_dir, "model_epoch%d.pth" % epoch))

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir):
        return net, optim, 0
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

## 4. 메인 실행 함수 (Mac Multiprocessing 방지용)
def main():
    # 파라미터 설정
    lr, batch_size, num_epochs = 1e-3, 4, 100
    data_dir, ckpt_dir, log_dir = './datasets', './checkpoints', './log'
    
    # Mac M1/M2 사용 시 'mps', 아니면 'cpu' (CUDA는 윈도우용)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로더
    transform = transforms.Compose([RandomFlip(), Normalization(), ToTensor()])
    dataset_train = AutumnDataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0) # Mac 에러 방지 위해 0 권장

    dataset_val = AutumnDataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # 네트워크 및 손실함수
    net = Unet().to(device)
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # 부수적인 설정
    num_batch_train = np.ceil(len(dataset_train) / batch_size)
    num_batch_val = np.ceil(len(dataset_val) / batch_size)
    
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean=0.5, std=0.5: (x*std) + mean
    fn_class = lambda x: 1.0*(x > 0.5)

    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    # 시작 에포크 로드
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    # 학습 루프
    for epoch in range(st_epoch + 1, num_epochs + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label, input = data['label'].to(device), data['input'].to(device)
            output = net(input)

            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]
            print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS %.4f' % 
                  (epoch, num_epochs, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard 기록
            if batch == 1: # 매 배치를 기록하면 느려지므로 첫 배치만 이미지 기록
                writer_train.add_image('label', fn_tonumpy(label), epoch, dataformats='NHWC')
                writer_train.add_image('input', fn_tonumpy(fn_denorm(input)), epoch, dataformats='NHWC')
                writer_train.add_image('output', fn_tonumpy(fn_class(output)), epoch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # 검증 루프
        with torch.no_grad():
            net.eval()
            loss_arr = []
            for batch, data in enumerate(loader_val, 1):
                label, input = data['label'].to(device), data['input'].to(device)
                output = net(input)
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]
                print('VAL  : EPOCH %04d/%04d | BATCH %04d/%04d | LOSS %.4f' % 
                      (epoch, num_epochs, batch, num_batch_val, np.mean(loss_arr)))

            writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 5 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

if __name__ == '__main__':
    main()