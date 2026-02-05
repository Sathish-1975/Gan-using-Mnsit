import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
import os

def save_images(generator, latent_dim, epoch, samples_dir="samples"):
    os.makedirs(samples_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        z = torch.randn(64, latent_dim)
        gen_imgs = generator(z)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Generated Images at Epoch {epoch}")
        plt.imshow(vutils.make_grid(gen_imgs, padding=2, normalize=True).permute(1, 2, 0))
        plt.savefig(f"{samples_dir}/epoch_{epoch}.png")
        plt.close()

def plot_losses(g_losses, d_losses, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
