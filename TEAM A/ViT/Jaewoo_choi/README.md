# Vision Transformer from Scratch

## Files

- `vit_paper_matched.py`: ViT implementation with concise paper-matching comments.
- `train_cifar10.py`: CIFAR-10 training script for Colab A100.
- `README.md`: summary and usage.

## Paper-to-Code Mapping

| Paper concept | Code |
|---|---|
| Image to patches | `PatchEmbedding` |
| Patch projection | `nn.Conv2d(kernel_size=patch_size, stride=patch_size)` |
| CLS token | `ViTEmbedding.cls_token` |
| Positional embedding | `ViTEmbedding.pos_embed` |
| Multi-head self-attention | `MultiHeadSelfAttention` |
| Transformer encoder block | `TransformerEncoderBlock` |
| MLP with GELU | `MLP` |
| Classification head | `ViTForImageClassification.head` |

## Quick Test

```bash
python vit_paper_matched.py
```

## CIFAR-10 Smoke Train

```bash
python train_cifar10.py --epochs 1 --batch-size 256 --train-samples 2048 --test-samples 512 --amp
```

## Full CIFAR-10 Train

```bash
python train_cifar10.py --epochs 50 --batch-size 256 --amp
```

## Pipeline

```text
image -> patches -> patch embeddings -> CLS token + positional embeddings -> Transformer encoder -> CLS classifier
```
