import torch
import torch.nn as nn
from utils import save_images

def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, lr, device):
    adversarial_loss = nn.BCELoss()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            
            # Ground truths
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(batch_size, latent_dim, device=device)
            
            # Generate a batch of images
            gen_imgs = generator(z)
            
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            
            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()

            if i % 400 == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        # Save images at the end of each epoch
        save_images(generator, latent_dim, epoch)

    return g_losses, d_losses
