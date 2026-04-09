## 라이브러리 추가하기
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

## 1. 네트워크 구축하기 (Unet 클래스는 기존과 동일하므로 생략하거나 그대로 두시면 됩니다)
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
        enc1_1 = self.enc1_1(x); enc1_2 = self.enc1_2(enc1_1); pool1 = self.pool1(enc1_2)
        enc2_1 = self.enc2_1(pool1); enc2_2 = self.enc2_2(enc2_1); pool2 = self.pool2(enc2_2)
        enc3_1 = self.enc3_1(pool2); enc3_2 = self.enc3_2(enc3_1); pool3 = self.pool3(enc3_2)
        enc4_1 = self.enc4_1(pool3); enc4_2 = self.enc4_2(enc4_1); pool4 = self.pool4(enc4_2)
        enc5_1 = self.enc5_1(pool4); dec5_1 = self.dec5_1(enc5_1)
        unpool4 = self.unpool4(dec5_1); cat4 = torch.cat((unpool4, enc4_2), dim=1); dec4_2 = self.dec4_2(cat4); dec4_1 = self.dec4_1(dec4_2)
        unpool3 = self.unpool3(dec4_1); cat3 = torch.cat((unpool3, enc3_2), dim=1); dec3_2 = self.dec3_2(cat3); dec3_1 = self.dec3_1(dec3_2)
        unpool2 = self.unpool2(dec3_1); cat2 = torch.cat((unpool2, enc2_2), dim=1); dec2_2 = self.dec2_2(cat2); dec2_1 = self.dec2_1(dec2_2)
        unpool1 = self.unpool1(dec2_1); cat1 = torch.cat((unpool1, enc1_2), dim=1); dec1_2 = self.dec1_2(cat1); dec1_1 = self.dec1_1(dec1_2)
        x = self.fc(dec1_1)
        return x

## 2. 데이터셋 및 트랜스폼 정의 (기존과 동일)
class AutumnDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        lst_data = os.listdir(self.data_dir)
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])
    def __len__(self): return len(self.lst_label)
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
        label, input = data['label'].transpose((2, 0, 1)).astype(np.float32), data['input'].transpose((2, 0, 1)).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): self.mean, self.std = mean, std
    def __call__(self, data):
        data['input'] = (data['input'] - self.mean) / self.std
        return data

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir) or not os.listdir(ckpt_dir): return net, optim, 0
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]), map_location='cpu')
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

## 3. 메인 실행 함수
def main():
    lr, batch_size = 1e-3, 4
    data_dir, ckpt_dir, result_dir = './datasets', './checkpoints', './results'

    # 폴더 생성 로직 보완
    for sub_dir in ['png', 'numpy']:
        path = os.path.join(result_dir, sub_dir)
        if not os.path.exists(path): os.makedirs(path)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([Normalization(), ToTensor()])
    dataset_test = AutumnDataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    net = Unet().to(device)
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    num_batch_test = np.ceil(len(dataset_test) / batch_size)
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean=0.5, std=0.5: (x*std) + mean
    fn_class = lambda x: 1.0*(x > 0.5)

    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_arr = []
        batch = 0 # 루프가 안 돌 경우를 대비해 초기화
        for batch, data in enumerate(loader_test, 1):
            label, input = data['label'].to(device), data['input'].to(device)
            output = net(input)
            loss = fn_loss(output, label)
            loss_arr += [loss.item()]
            
            print('TEST : BATCH %04d/%04d | LOSS %.4f' % (batch, num_batch_test, np.mean(loss_arr)))

            label_np = fn_tonumpy(label)
            input_np = fn_tonumpy(fn_denorm(input))
            output_np = fn_tonumpy(fn_class(output))

            for j in range(label_np.shape[0]):
                idx = int((batch - 1) * batch_size + j)
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % idx), label_np[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % idx), input_np[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % idx), output_np[j].squeeze(), cmap='gray')
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % idx), label_np[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % idx), input_np[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % idx), output_np[j].squeeze())

        # 이 print 문을 main 함수 내부(루프 바깥)로 들여쓰기 했습니다.
        if batch > 0:
            print('AVERAGE : BATCH %04d/%04d | LOSS %.4f' % (batch, num_batch_test, np.mean(loss_arr)))
        else:
            print("테스트 데이터가 없습니다. 경로를 확인해주세요.")

if __name__ == '__main__':
    main()