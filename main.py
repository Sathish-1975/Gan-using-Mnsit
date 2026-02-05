import torch
from model import Generator, Discriminator
from dataset import get_dataloader
from train import train_gan
from utils import plot_losses

def main():
    # Parameters
    latent_dim = 100
    num_epochs = 10 # 10 epochs for visible results
    batch_size = 64
    lr = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Initialize models
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)

    # Initialize dataloader
    dataloader = get_dataloader(batch_size=batch_size)

    # Start training
    g_losses, d_losses = train_gan(
        generator, 
        discriminator, 
        dataloader, 
        latent_dim, 
        num_epochs, 
        lr, 
        device
    )

    # Plot results
    plot_losses(g_losses, d_losses)
    
    # Save the models
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Training finished. Results saved in 'samples/', 'loss_curve.png', 'generator.pth', and 'discriminator.pth'.")

if __name__ == "__main__":
    main()
