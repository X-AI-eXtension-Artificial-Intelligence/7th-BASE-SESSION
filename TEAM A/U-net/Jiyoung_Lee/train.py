import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Dataset, ToTensor, Normalization
from model import UNet

from torchvision import transforms

# =========================
# 설정
# =========================
lr = 1e-3
batch_size = 2
num_epoch = 5

data_dir = './datasets'
ckpt_dir = './saved_model'
result_dir = './result'

os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 데이터 로드
# =========================
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

dataset_train = Dataset(os.path.join(data_dir, 'train'), transform=transform)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# =========================
# 모델
# =========================
net = UNet().to(device)

fn_loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# =========================
# 저장 폴더 생성
# =========================
os.makedirs(ckpt_dir, exist_ok=True)

# =========================
# 학습
# =========================
for epoch in range(num_epoch):
    net.train()
    loss_arr = []

    for data in loader_train:
        label = data['label'].to(device)
        input = data['input'].to(device)

        optim.zero_grad()

        output = net(input)
        loss = fn_loss(output, label)

        loss.backward()
        optim.step()

        loss_arr.append(loss.item())

    print(f"EPOCH {epoch+1} | LOSS {np.mean(loss_arr):.4f}")

    with torch.no_grad():
        net.eval()
        for batch, data in enumerate(loader_train):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # numpy 변환
            input_np = input.cpu().numpy()[0,0]
            label_np = label.cpu().numpy()[0,0]
            output_np = (torch.sigmoid(output).cpu().numpy()[0,0] > 0.5).astype(np.float32)

            np.save(os.path.join(result_dir, 'numpy', f'input_{batch}.npy'), input_np)
            np.save(os.path.join(result_dir, 'numpy', f'label_{batch}.npy'), label_np)
            np.save(os.path.join(result_dir, 'numpy', f'output_{batch}.npy'), output_np)

            plt.imsave(os.path.join(result_dir, 'png', f'input_{batch}.png'), input_np, cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'label_{batch}.png'), label_np, cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{batch}.png'), output_np, cmap='gray')

            break  # 한 장만 저장

    # 저장
    torch.save(net.state_dict(), os.path.join(ckpt_dir, f"model_epoch{epoch+1}.pth"))

print("학습 완료")