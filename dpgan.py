
import math
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root='./data',
                          train=True,
                          download=True,
                          transform=transform)

print(train_ds.data.shape)
print(train_ds.targets.shape)
print(train_ds.classes)
print(train_ds.data[0])
print(train_ds.targets[0])
print(train_ds.data[0].max())
print(train_ds.data[0].min())
print(train_ds.data[0].float().mean())
print(train_ds.data[0].float().std())



# Loaddata
dl = DataLoader(dataset=train_ds,
                shuffle=True,
                batch_size=64)

# Examine a sample batch from the dataloader
image_batch = next(iter(dl))
print(len(image_batch), type(image_batch))
print(image_batch[0].shape)
print(image_batch[1].shape)

## Visualise a sample batch

def display_images(images, n_cols=4, figsize=(12, 6)):
    plt.style.use('ggplot')
    n_images = len(images)
    n_rows = math.ceil(n_images / n_cols)
    plt.figure(figsize=figsize)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx+1)
        image = images[idx]
        # make dims H x W x C
        image = image.permute(1, 2, 0)
        cmap = 'gray' if image.shape[2] == 1 else plt.cm.viridis
        ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

display_images(images=image_batch[0], n_cols=8)

class Discriminator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Discriminator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=128)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=32, out_features=out_features)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.leaky_relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.leaky_relu3(x)
        x = self.dropout(x)
        logit_out = self.fc4(x)

        return logit_out
class Generator(nn.Module):
     def __init__(self, in_features, out_features):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.fc4 = nn.Linear(in_features=128, out_features=out_features)
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

     def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)

        return tanh_out
def real_loss(predicted_outputs, loss_fn, device, epsilon=1000):
    batch_size = predicted_outputs.shape[0]
    targets = torch.ones(batch_size).to(device)
    real_loss = loss_fn(predicted_outputs.squeeze(), targets)

    return real_loss

def fake_loss(predicted_outputs, loss_fn, device, epsilon=1000):
    batch_size = predicted_outputs.shape[0]
    targets = torch.zeros(batch_size).to(device)
    fake_loss = loss_fn(predicted_outputs.squeeze(), targets)

    return fake_loss

# Sample generation of latent vector
z_size = 100
z = np.random.uniform(-1, 1, size=(16, z_size))
plt.imshow(z, cmap='gray')

plt.xticks([])
plt.yticks([])
# plt.show()

# Training loop function
def train_minst_gan(d, g, d_optim, g_optim, loss_fn, dl, n_epochs, device, epsilon=1000, verbose=False):
    print(f'Training on [{device}]...')

    fixed_z = torch.randn(16, z_size, device=device)

    d_losses = []
    g_losses = []

    d = d.to(device)
    g = g.to(device)

    for epoch in range(n_epochs):
        print(f'Epoch [{epoch+1}/{n_epochs}]:')
        d.train()
        g.train()

        d_running_loss = 0.0
        g_running_loss = 0.0

        for batch_idx, (real_images, _) in enumerate(dl):
            real_images = real_images.to(device)

            ## Train discriminator using real and then fake MNIST images,
            d_optim.zero_grad()

            # Real MNIST images
            real_logits = d(real_images)
            d_real_loss = real_loss(real_logits, loss_fn, device, epsilon=epsilon)

            # Fake images
            with torch.no_grad():
                z = torch.randn(real_images.size(0), z_size, device=device)
                fake_images = g(z)

            fake_logits = d(fake_images)
            d_fake_loss = fake_loss(fake_logits, loss_fn, device, epsilon=epsilon)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            d_running_loss += d_loss.item()

            # Reset gradients
            g_optim.zero_grad()

            z = torch.randn(real_images.size(0), z_size, device=device)
            fake_images = g(z)

            fake_logits = d(fake_images)
            g_loss = real_loss(fake_logits, loss_fn, device, epsilon=epsilon)

            g_loss.backward()
            g_optim.step()

            g_running_loss += g_loss.item()

            if verbose and batch_idx % 200 == 0:
                print(f'\tBatch [{batch_idx}/{len(dl)}] - D Loss: {d_loss.item():.6f}, G Loss: {g_loss.item():.6f}')

        d_losses.append(d_running_loss / len(dl))
        g_losses.append(g_running_loss / len(dl))

        print(f'\tDiscriminator Loss: {d_losses[-1]}, Generator Loss: {g_losses[-1]}')

    return d_losses, g_losses

# Instantiate Discriminator and Generator
d = Discriminator(in_features=784, out_features=1)
g = Generator(in_features=100, out_features=784)
print(d)
print(g)

# Instantiate optimizers
d_optim = optim.Adam(d.parameters(), lr=0.002)
g_optim = optim.Adam(g.parameters(), lr=0.002)

# Instantiate the loss function
loss_fn = nn.BCEWithLogitsLoss()

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train
n_epochs = 1000
d_losses, g_losses = train_minst_gan(d, g, d_optim, g_optim, loss_fn, dl, n_epochs, device, verbose=False)
##
## Visualize training losses
##
plt.plot(d_losses, label='Discriminator')
plt.plot(g_losses, label='Generator')
plt.legend()
plt.show()

def show_generated_images(epoch, generator, device, n_cols=8, z_size=100):
    generator.eval()
    z = torch.randn(64, z_size, device=device)
    fake_images = generator(z)
    fake_images = fake_images.view(-1, 1, 28, 28).cpu().detach()

    display_images(fake_images, n_cols=n_cols)

# Display generated images for different epochs
show_generated_images(epoch=50, generator=g, device=device, n_cols=8)

show_generated_images(epoch=100, generator=g, device=device, n_cols=8)
