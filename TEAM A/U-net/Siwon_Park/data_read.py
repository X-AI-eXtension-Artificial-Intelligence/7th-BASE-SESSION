"""
데이터 전처리 스크립트
tif 멀티프레임 → train / val / test npy 분할 저장
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=str, default='./datasets')
    p.add_argument('--name_label',  type=str, default='train-labels.tif')
    p.add_argument('--name_input',  type=str, default='train-volume.tif')
    p.add_argument('--n_train',     type=int, default=24)
    p.add_argument('--n_val',       type=int, default=3)
    p.add_argument('--n_test',      type=int, default=3)
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    label_path = os.path.join(args.data_dir, args.name_label)
    input_path = os.path.join(args.data_dir, args.name_input)

    assert os.path.exists(label_path), f"label 파일 없음: {label_path}"
    assert os.path.exists(input_path), f"input 파일 없음: {input_path}"

    img_label = Image.open(label_path)
    img_input = Image.open(input_path)
    nframe = img_label.n_frames

    need = args.n_train + args.n_val + args.n_test
    assert nframe >= need, f"프레임 수({nframe})가 필요량({need})보다 적습니다."

    # 재현 가능한 셔플
    np.random.seed(args.seed)
    idx = np.random.permutation(nframe)

    splits = {
        'train': idx[:args.n_train],
        'val':   idx[args.n_train: args.n_train + args.n_val],
        'test':  idx[args.n_train + args.n_val: need],
    }

    for split, frame_ids in splits.items():
        save_dir = os.path.join(args.data_dir, split)
        os.makedirs(save_dir, exist_ok=True)
        for i, fid in enumerate(frame_ids):
            img_label.seek(fid)
            img_input.seek(fid)
            np.save(os.path.join(save_dir, f'label_{i:03d}.npy'), np.asarray(img_label))
            np.save(os.path.join(save_dir, f'input_{i:03d}.npy'), np.asarray(img_input))
        print(f"[{split}] {len(frame_ids)}장 저장 → {save_dir}")

    # 시각화: train 첫 번째 샘플
    img_label.seek(splits['train'][0])
    img_input.seek(splits['train'][0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.asarray(img_input), cmap='gray'); axes[0].set_title('Input')
    axes[1].imshow(np.asarray(img_label), cmap='gray'); axes[1].set_title('Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.data_dir, 'sample_preview.png'), dpi=120)
    print("샘플 시각화 저장 완료")


if __name__ == '__main__':
    main()
