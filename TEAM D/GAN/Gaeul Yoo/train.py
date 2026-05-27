import os
import torch
import torch.nn as nn
from torchvision.utils import save_image


def train(G, D, dataloader, epochs, latent_dim, lr, device, output_dir):
    """
    GAN training loop — Goodfellow 2014, Algorithm 1
    k = 1 (one D step per G step)

    D loss: BCELoss(D(real), 1) + BCELoss(D(G(z).detach()), 0)
    G loss: BCELoss(D(G(z)), 1)   ← non-saturating variant (Goodfellow Section 3)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    criterion = nn.BCELoss()
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # fixed noise: same z every epoch so we can track visual progress
    fixed_z = torch.randn(64, latent_dim, device=device)

    d_losses, g_losses = [], []

    for epoch in range(epochs):
        d_epoch, g_epoch = 0.0, 0.0

        for real_imgs, _ in dataloader:
            batch = real_imgs.size(0)
            real_imgs = real_imgs.view(batch, -1).to(device)   # flatten to (B, 784)

            real_labels = torch.ones(batch, 1, device=device)
            fake_labels = torch.zeros(batch, 1, device=device)

            # ── Discriminator step ───────────────────────────────────────────
            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = G(z)                                    # no detach yet

            real_loss = criterion(D(real_imgs), real_labels)
            fake_loss = criterion(D(fake_imgs.detach()), fake_labels)  # detach G graph
            d_loss = real_loss + fake_loss

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ── Generator step ───────────────────────────────────────────────
            z = torch.randn(batch, latent_dim, device=device)
            fake_imgs = G(z)
            g_loss = criterion(D(fake_imgs), real_labels)      # fool D → label = 1

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            d_epoch += d_loss.item()
            g_epoch += g_loss.item()

        d_avg = d_epoch / len(dataloader)
        g_avg = g_epoch / len(dataloader)
        d_losses.append(d_avg)
        g_losses.append(g_avg)

        print(f"Epoch [{epoch+1:03d}/{epochs}]  D_loss: {d_avg:.4f}  G_loss: {g_avg:.4f}")

        # ── Save sample grid every 10 epochs ────────────────────────────────
        if (epoch + 1) % 10 == 0:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z).view(64, 1, 28, 28)
                samples = (samples + 1) / 2          # [-1,1] → [0,1] for saving
                save_image(
                    samples,
                    os.path.join(output_dir, f"samples_epoch_{epoch+1:04d}.png"),
                    nrow=8,
                )
            G.train()

        # ── Save checkpoints every 50 epochs ────────────────────────────────
        if (epoch + 1) % 50 == 0:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch_{epoch+1:04d}.pth"))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch_{epoch+1:04d}.pth"))
            print(f"  → Checkpoint saved at epoch {epoch+1}")

    # ── Final weights ────────────────────────────────────────────────────────
    torch.save(G.state_dict(), os.path.join(output_dir, "G_final.pth"))
    torch.save(D.state_dict(), os.path.join(output_dir, "D_final.pth"))
    print("Training complete. Final weights saved.")

    return d_losses, g_losses
