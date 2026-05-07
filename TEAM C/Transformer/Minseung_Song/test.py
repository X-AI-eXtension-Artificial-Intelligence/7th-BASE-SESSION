"""
test.py
- 학습된 Transformer 체크포인트를 불러와 Multi30k test set에서 추론 수행
- (1) 샘플 N개 번역 예시 출력, (2) 전체 test set BLEU 점수 계산

사용 예시:
    python test.py                      # saved/ 안에서 가장 작은 valid loss 체크포인트 자동 선택
    python test.py --checkpoint saved/model-3.20.pt
    python test.py --num_samples 10
"""
import argparse
import glob
import os

import torch

from conf import *
from data import *
from models.model.transformer import Transformer
from util.bleu import get_bleu, idx_to_word


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


def evaluate_bleu(model, iterator):
    """train.py의 evaluate와 동일한 방식으로 BLEU 점수 계산."""
    model.eval()
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])

            total_bleu = []
            cur_batch = src.size(0)
            for j in range(cur_batch):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(
                        hypotheses=output_words.split(),
                        reference=trg_words.split(),
                    )
                    total_bleu.append(bleu)
                except Exception:
                    pass

            if total_bleu:
                batch_bleu.append(sum(total_bleu) / len(total_bleu))

    if not batch_bleu:
        return 0.0
    return sum(batch_bleu) / len(batch_bleu)


def show_samples(model, iterator, num_samples=5):
    """test set에서 num_samples 개를 뽑아 src/정답/예측을 출력."""
    model.eval()
    shown = 0
    print("\n===== Sample Translations (EN -> DE) =====")
    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            pred = output.max(dim=2)[1]

            cur_batch = src.size(0)
            for j in range(cur_batch):
                if shown >= num_samples:
                    return
                src_words = idx_to_word(src[j], loader.source.vocab)
                trg_words = idx_to_word(trg[j], loader.target.vocab)
                pred_words = idx_to_word(pred[j], loader.target.vocab)
                print(f"\n[{shown + 1}]")
                print(f"  SRC : {src_words}")
                print(f"  TRG : {trg_words}")
                print(f"  PRED: {pred_words}")
                shown += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="체크포인트 경로. 비우면 saved/ 안에서 가장 좋은 모델 자동 선택")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="출력할 번역 예시 개수")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or find_best_checkpoint()
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    print(f"[INFO] Device: {device}")

    model = build_model()
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    show_samples(model, test_iter, num_samples=args.num_samples)

    print("\n[INFO] Computing BLEU on full test set...")
    bleu_score = evaluate_bleu(model, test_iter)
    print(f"\n===== Test BLEU: {bleu_score:.3f} =====")


if __name__ == "__main__":
    main()
