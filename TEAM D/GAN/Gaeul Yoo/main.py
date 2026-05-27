import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import Generator, Discriminator
from train import train
from utils import plot_losses


def get_args():
    p = argparse.ArgumentParser(description="GAN — Goodfellow 2014 (MNIST)")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=0.0002)
    p.add_argument("--latent_dim",  type=int,   default=100)
    p.add_argument("--output_dir",  type=str,   default="outputs")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}  |  z_dim: {args.latent_dim}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    # Normalize to [-1, 1] to match Generator's Tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # (mean), (std) → [-1, 1]
    ])
    dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(latent_dim=args.latent_dim).to(device)
    D = Discriminator().to(device)

    print("\n[Generator]")
    print(G)
    print("\n[Discriminator]")
    print(D)

    total_G = sum(p.numel() for p in G.parameters())
    total_D = sum(p.numel() for p in D.parameters())
    print(f"\nG params: {total_G:,}  |  D params: {total_D:,}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    d_losses, g_losses = train(
        G, D, dataloader,
        epochs=args.epochs,
        latent_dim=args.latent_dim,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_losses(d_losses, g_losses, args.output_dir)


if __name__ == "__main__":
    main()
