# 필수 라이브러리 임포트
import argparse  # 명령행 인자 파싱
import os  # 파일 및 디렉토리 조작
import numpy as np  # 수치 연산
import torch  # PyTorch 핵심
import torch.nn as nn  # 신경망 관련 모듈
from torch.utils.data import DataLoader  # 데이터 로더
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 기록

# 사용자 정의 모듈 import
from model import UNet  # U-Net 모델 구조
from dataset import *  # 데이터셋, 트랜스폼 정의
from util import *  # 모델 저장/불러오기 함수

import matplotlib.pyplot as plt  # 이미지 저장용
from torchvision import transforms  # 이미지 전처리 도구

def main():
    # 명령행 인자 정의
    parser = argparse.ArgumentParser(description="Train the UNet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--lr", default=1e-3, type=float, dest="lr")  # 학습률
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")  # 배치 크기
    parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")  # 에폭 수
    parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")  # 데이터 경로
    parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")  # 체크포인트 저장 경로
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")  # 로그 저장 경로
    parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")  # 테스트 결과 저장 경로
    parser.add_argument("--mode", default="train", type=str, dest="mode")  # 실행 모드 (train/test)
    parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")  # 이어서 학습 여부

    args = parser.parse_args()  # 인자 파싱

    # 인자 변수로 분리
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    result_dir = args.result_dir
    mode = args.mode
    train_continue = args.train_continue

    # CUDA 사용 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 설정 출력
    print("learning rate: %.4e" % lr)
    print("batch size: %d" % batch_size)
    print("number of epoch: %d" % num_epoch)
    print("data dir: %s" % data_dir)
    print("ckpt dir: %s" % ckpt_dir)
    print("log dir: %s" % log_dir)
    print("result dir: %s" % result_dir)
    print("mode: %s" % mode)

    # 결과 디렉토리 없으면 생성
    if not os.path.exists(result_dir):
        os.makedirs(os.path.join(result_dir, 'png'))
        os.makedirs(os.path.join(result_dir, 'numpy'))

    # 데이터셋 로딩
    if mode == 'train':
        # 학습용 트랜스폼
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
        # 학습/검증 데이터셋 및 로더
        dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
        dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
        num_data_train = len(dataset_train)
        num_data_val = len(dataset_val)
        num_batch_train = np.ceil(num_data_train / batch_size)
        num_batch_val = np.ceil(num_data_val / batch_size)
    else:
        # 테스트용 트랜스폼
        transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
        dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
        loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
        num_data_test = len(dataset_test)
        num_batch_test = np.ceil(num_data_test / batch_size)

    # 모델, 손실함수, 옵티마이저 정의
    net = UNet().to(device)
    fn_loss = nn.BCEWithLogitsLoss().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    # 기타 부수 함수들
    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
    fn_denorm = lambda x, mean, std: (x * std) + mean
    fn_class = lambda x: 1.0 * (x > 0.5)

    # TensorBoard 로그 기록기
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    st_epoch = 0  # 시작 에폭

    # ---------------------- TRAIN ----------------------
    if mode == 'train':
        if train_continue == "on":
            # 체크포인트 불러오기
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        # 에폭 루프
        for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            loss_arr = []

            # 배치 루프
            for batch, data in enumerate(loader_train, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)
                output = net(input)
                optim.zero_grad()
                loss = fn_loss(output, label)
                loss.backward()
                optim.step()
                loss_arr += [loss.item()]

                # 학습 로그 출력
                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                # 이미지 TensorBoard 기록
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))
                writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
                writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

            # 에폭당 평균 손실 기록
            writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

            # ---------- VALID ----------
            with torch.no_grad():
                net.eval()
                loss_arr = []
                for batch, data in enumerate(loader_val, 1):
                    label = data['label'].to(device)
                    input = data['input'].to(device)
                    output = net(input)
                    loss = fn_loss(output, label)
                    loss_arr += [loss.item()]

                    # 검증 로그 출력
                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                          (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                    # 이미지 TensorBoard 기록
                    label = fn_tonumpy(label)
                    input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                    output = fn_tonumpy(fn_class(output))
                    writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                    writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

                writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            # 주기적으로 모델 저장
            if epoch % 50 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

        # 로그 종료
        writer_train.close()
        writer_val.close()

    # ---------------------- TEST ----------------------
    else:
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
                    plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                    plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')
                    np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                    np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                    np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

            print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

# Windows에서 multiprocessing 사용을 위해 필요
if __name__ == '__main__':
    main()
