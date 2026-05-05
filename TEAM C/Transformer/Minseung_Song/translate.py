"""
translate.py
- 학습된 Transformer로 영어 → 독일어 인터랙티브 번역
- test.py가 teacher forcing 평가였다면, 이 스크립트는 진짜 추론(autoregressive greedy decoding)
- 직접 영문장을 입력하면 모델이 한 토큰씩 자기 손으로 독일어를 뽑아낸다.

사용:
    python translate.py                              # saved/ 안에서 가장 좋은 체크포인트 자동 선택
    python translate.py --checkpoint saved/model-3.20.pt
    python translate.py --max_len 50                 # 번역 최대 길이 토큰 수
"""
import argparse
import glob
import os

import torch

from conf import *
from data import *  # tokenizer, loader, src_pad_idx, trg_pad_idx, trg_sos_idx, ...
from models.model.transformer import Transformer


def find_best_checkpoint(saved_dir="saved"):
    """saved/model-<loss>.pt 패턴 중 loss가 가장 작은 파일을 반환."""
    files = glob.glob(os.path.join(saved_dir, "model-*.pt"))
    if not files:
        raise FileNotFoundError(
            f"'{saved_dir}'에 학습된 체크포인트(model-*.pt)가 없습니다. 먼저 train.py를 돌려주세요."
        )

    def parse_loss(fname):
        try:
            return float(os.path.basename(fname).replace("model-", "").replace(".pt", ""))
        except ValueError:
            return float("inf")

    return min(files, key=parse_loss)


def build_model():
    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_sos_idx=trg_sos_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)
    return model


def tokenize_and_encode(sentence):
    """영어 문장 -> spaCy 토큰화 -> 소문자 -> source vocab 인덱스."""
    tokens = [tok.lower() for tok in tokenizer.tokenize_en(sentence)]
    tokens = [loader.source.init_token] + tokens + [loader.source.eos_token]
    indices = [loader.source.vocab.stoi[t] for t in tokens]
    return indices, tokens


def translate(model, sentence, max_decode_len=100):
    """입력 영어 문장 -> 독일어 번역 결과 (autoregressive greedy)."""
    model.eval()

    # 1. encoder
    src_indices, src_tokens = tokenize_and_encode(sentence)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)  # [1, src_len]
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 2. autoregressive greedy decoding
    trg_eos_idx = loader.target.vocab.stoi[loader.target.eos_token]
    trg_indices = [trg_sos_idx]

    for _ in range(max_decode_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)  # [1, cur_len]
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        next_token = output.argmax(dim=2)[:, -1].item()
        trg_indices.append(next_token)
        if next_token == trg_eos_idx:
            break

    # 3. indices -> 단어 (특수 토큰 스킵)
    skip = {
        loader.target.init_token,
        loader.target.eos_token,
        '<pad>',
        '<unk>',
    }
    words = []
    for idx in trg_indices:
        word = loader.target.vocab.itos[idx]
        if word in skip:
            continue
        words.append(word)

    return " ".join(words), src_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로. 비우면 saved/ 안에서 가장 좋은 모델 자동 선택")
    parser.add_argument("--max_len", type=int, default=100,
                        help="번역 최대 길이 (토큰 수)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_best_checkpoint()
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")

    model = build_model()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("\n번역할 영어 문장을 입력하세요. (quit / exit / 빈 줄로 종료)\n")
    while True:
        try:
            sentence = input("EN > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not sentence or sentence.lower() in ("quit", "exit", ":q"):
            print("종료합니다.")
            break

        try:
            translation, src_tokens = translate(model, sentence, max_decode_len=args.max_len)
            print(f"   (tokens: {' '.join(src_tokens)})")
            print(f"DE > {translation}\n")
        except Exception as e:
            print(f"   [에러] {e}\n")


if __name__ == "__main__":
    main()
