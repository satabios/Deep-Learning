import ipdb
import torch
import torchvision

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import numpy as np
import matplotlib.pyplot as plt

import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.utils import make_grid

seed_everything(42, workers=True)
device ="cuda"



cgan=True
batch_size = 4*8
num_classes = 10
image_size = 28
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

def one_hot(data):
    encoding  = np.zeros((len(data), num_classes))
    for ind in range(len(data)):
        encoding[ind,data[ind]-1] = 1
    return encoding

def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./dataset/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./dataset/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size)



#Formula for ConvTranspose:
# Output Shape = (n-1) * (s) - (2p) + (dilation) * (f -1) + (o/p padding) + 1
image_channels = 1
latent_size = 128
gen_latent_size = latent_size+ num_classes if cgan == True else latent_size
discriminator_channel = image_channels + num_classes if cgan == True else image_channels

generator = nn.Sequential(
    # in: latent_size x 1 x 1

    nn.ConvTranspose2d(gen_latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    # out: 512 x 4 x 4

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    # out: 256 x 8 x 8

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    # out: 128 x 16 x 16


    nn.ConvTranspose2d(128, 1, kernel_size=2, stride=2, padding=2, bias=False),
    nn.Tanh()
    # out: 1 x 28 x 28
).to(device)



class red_dim(nn.Module):
    def forward(self, input):
        return torch.squeeze(input, axis=-1)
reduce_dim = red_dim()

discriminator = nn.Sequential(
    # in: 1 x 64 x 64

    nn.Conv2d(discriminator_channel, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 64 x 32 x 32

    nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 128 x 16 x 16

    nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 256 x 8 x 8

    nn.Conv2d(64, 32, kernel_size=4, stride=1, padding=0, bias=False),
    nn.BatchNorm2d(32),
    nn.LeakyReLU(0.2, inplace=True),
    # out: 512 x 4 x 4

    nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0, bias=False),
    # out: 1 x 1 x 1

    nn.Flatten(),
    # red_dim(),
    nn.Sigmoid()).to(device)


def train_discriminator(real_images, opt_d, labels):
    one_hot_labels = torch.tensor(one_hot(labels), device=device)
    # Clear discriminator gradients
    opt_d.zero_grad()

    latent = torch.randn(batch_size, latent_size, device=device)
    if(cgan==True):
        # label_embeddings = nn.Embedding(10,10)(labels).unsqueeze(2).unsqueeze(3).to(device)
        latent = torch.cat((latent, one_hot_labels), 1).float()
        one_hot_image_labels = one_hot_labels.repeat( image_size,image_size)
        one_hot_image_labels = one_hot_image_labels.view(-1, num_classes, image_size, image_size)
        real_images = torch.cat((real_images, one_hot_image_labels),1).float()


    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images


    fake_images = generator(latent.unsqueeze(2).unsqueeze(3))

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    if(cgan == True):
        fake_images = torch.cat((fake_images, one_hot_image_labels),1).float()

    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g,labels):
    one_hot_labels = torch.tensor(one_hot(labels)).to(device)
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, latent_size,  device=device)
    if(cgan==True):
        latent = torch.cat((latent, one_hot_labels), 1).float()
        one_hot_image_labels = one_hot_labels.repeat( image_size,image_size)
        one_hot_image_labels = one_hot_image_labels.view(-1, num_classes, image_size, image_size)

    fake_images = generator(latent.unsqueeze(2).unsqueeze(3))


    if(cgan==True):
        fake_images = torch.cat((fake_images, one_hot_image_labels),1).float()
    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()






sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

def save_samples(index, latent_tensors, show=True):

    if(cgan==True):
        labels = torch.from_numpy(np.array([num for _ in range(8) for num in range(8)]))#.astype(torch.cuda.LongTensor)
        label_embeddings = nn.Embedding(10,10)(labels).unsqueeze(2).unsqueeze(3).to(device)
        latent_tensors = torch.cat((latent_tensors, label_embeddings), 1)

    fake_images = generator(latent_tensors).to(device)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)

    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, labels in tqdm(train_loader):
            # Train discriminator
            real_images = real_images.to(device)
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d, labels)
            # Train generator

            loss_g = train_generator(opt_g, labels)


        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # fixed_latent.get_device()
        # generator.get_device()
        # ipdb.set_trace()
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores


lr = 0.0002
epochs = 60

history = fit(epochs, lr)


# TODO: * Try Other Types of GAN
#        * Conditonal GAN
#       * Create global functions for data loaders
