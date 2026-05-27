"""
Evaluation script for trained ViT models.

Usage:
    # 기본 정확도 평가
    python evaluate.py --exp-name vit-test

    # 특정 체크포인트로 평가
    python evaluate.py --exp-name vit-test --checkpoint model_50.pt

    # 클래스별 정확도까지 출력
    python evaluate.py --exp-name vit-test --per-class

    # Attention map 시각화까지 저장
    python evaluate.py --exp-name vit-test --visualize
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from data import prepare_data
from utils import load_experiment, visualize_attention


CLASSES = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


@torch.no_grad()
def evaluate(model, testloader, device):
    """전체 정확도와 loss 계산."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    for batch in testloader:
        batch = [t.to(device) for t in batch]
        images, labels = batch
        logits, _ = model(images)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(images)
        predictions = torch.argmax(logits, dim=1)
        correct += torch.sum(predictions == labels).item()
        total += len(images)
    accuracy = correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


@torch.no_grad()
def evaluate_per_class(model, testloader, device, num_classes=10):
    """클래스별 정확도 계산."""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for batch in testloader:
        batch = [t.to(device) for t in batch]
        images, labels = batch
        logits, _ = model(images)
        predictions = torch.argmax(logits, dim=1)
        for label, pred in zip(labels, predictions):
            class_total[label.item()] += 1
            if label == pred:
                class_correct[label.item()] += 1
    per_class_acc = [
        (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)
        for i in range(num_classes)
    ]
    return per_class_acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained ViT model on CIFAR-10")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="실험 이름 (experiments/<exp-name>/ 아래에서 모델 로드)")
    parser.add_argument("--checkpoint", type=str, default="model_final.pt",
                        help="로드할 체크포인트 파일명 (기본: model_final.pt)")
    parser.add_argument("--base-dir", type=str, default="experiments",
                        help="실험 결과 저장 디렉토리")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda 또는 cpu (기본: auto)")
    parser.add_argument("--per-class", action="store_true",
                        help="클래스별 정확도 출력")
    parser.add_argument("--visualize", action="store_true",
                        help="Attention map을 시각화해서 PNG로 저장")
    parser.add_argument("--vis-output", type=str, default=None,
                        help="시각화 저장 경로 (기본: experiments/<exp-name>/attention.png)")
    parser.add_argument("--save-results", action="store_true",
                        help="평가 결과를 JSON으로 저장")
    return parser.parse_args()


def main():
    args = parse_args()

    # Device 설정
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    if device == "cuda" and torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # 모델 로드
    print(f"[Loading] experiments/{args.exp_name}/{args.checkpoint}")
    config, model, train_losses, test_losses, accuracies = load_experiment(
        args.exp_name,
        checkpoint_name=args.checkpoint,
        base_dir=args.base_dir,
    )
    model = model.to(device)
    print(f"[Model] {count_parameters(model):,} trainable parameters")
    print(f"[Config] {json.dumps(config, indent=2)}")

    # 학습 히스토리 요약
    if accuracies:
        best_epoch = int(max(range(len(accuracies)), key=lambda i: accuracies[i]))
        print(f"[Training history] {len(accuracies)} epochs trained")
        print(f"  Best val accuracy: {accuracies[best_epoch]:.4f} (epoch {best_epoch + 1})")
        print(f"  Final val accuracy: {accuracies[-1]:.4f}")

    # 데이터 로드
    print("[Data] loading CIFAR-10 test set...")
    _, testloader, _ = prepare_data(batch_size=args.batch_size, num_workers=2)

    # 전체 정확도
    print("[Evaluating] running on test set...")
    accuracy, loss = evaluate(model, testloader, device)
    print(f"\n=== Overall Result ===")
    print(f"  Test loss     : {loss:.4f}")
    print(f"  Test accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 클래스별 정확도
    per_class = None
    if args.per_class:
        per_class = evaluate_per_class(model, testloader, device, num_classes=len(CLASSES))
        print(f"\n=== Per-Class Accuracy ===")
        for name, acc in zip(CLASSES, per_class):
            print(f"  {name:<8s} : {acc:.4f} ({acc * 100:.2f}%)")

    # Attention 시각화
    if args.visualize:
        vis_output = args.vis_output
        if vis_output is None:
            outdir = os.path.join(args.base_dir, args.exp_name)
            vis_output = os.path.join(outdir, "attention.png")
        print(f"\n[Visualizing] attention maps -> {vis_output}")
        # matplotlib을 화면 없이 사용하려면 백엔드 변경
        import matplotlib
        matplotlib.use("Agg")
        visualize_attention(model, output=vis_output, device=device)
        print(f"  saved.")

    # 결과 JSON 저장
    if args.save_results:
        outdir = os.path.join(args.base_dir, args.exp_name)
        os.makedirs(outdir, exist_ok=True)
        results_file = os.path.join(outdir, "evaluation.json")
        results = {
            "checkpoint": args.checkpoint,
            "test_loss": loss,
            "test_accuracy": accuracy,
            "num_parameters": count_parameters(model),
        }
        if per_class is not None:
            results["per_class_accuracy"] = {
                cls: acc for cls, acc in zip(CLASSES, per_class)
            }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved] evaluation results -> {results_file}")


if __name__ == "__main__":
    main()
