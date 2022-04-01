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

seed_everything(42, workers=True)

batch_size = 8*8*8
num_classes = 10
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



def one_hot(data):
    encoding  = np.zeros((len(data), num_classes))
    for ind in range(len(data)):
        encoding[ind,data[ind]-1] = 1
    return encoding

class autoencoder(pl.LightningModule):
    def __init__(self, vae):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 4 * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(4 * 2, 4, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(4, 1, 2, stride=2)

        # self.flatten =
        self.fc1 = nn.Linear(392, 16)
        self.fc2 = nn.Linear(16, num_classes)
        self.vae = vae
        self.mu = nn.Linear(392, 16)
        self.sigma = nn.Linear(392, 16)
        self.fc3  = nn.Linear(16, 392)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        embedding = x.view(-1, 392)
        embedding_out = F.relu(self.fc1(embedding))
        embedding = self.fc2(embedding_out)
        if(self.vae ==1):
            mu = self.mu(x)
            sigma = torch.exp(self.sigma(x))
            x = mu + sigma * self.N.sample(mu.shape)
            self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()



        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))

        return x, embedding_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = torch.from_numpy(one_hot(y)).to("cuda")
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        embedding = out.view(-1,392)
        embedding_out = F.relu(self.fc1(embedding))
        embedding = self.fc2(embedding_out)
        if (self.vae == 1):
            out = out.view(-1, 392)
            mu = self.mu(out)
            sigma = torch.exp(self.sigma(out))
            out = mu + sigma * self.N.sample(mu.shape)
            out = self.fc3(out)
            out = out.view(-1, 8, 7, 7)
            self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()

        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        if (self.vae == 1):
            loss = ((x - out) ** 2).sum() + self.kl

        else:
            loss = F.mse_loss(out, x) + F.cross_entropy(embedding, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = torch.from_numpy(one_hot(y)).to("cuda")
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        # 8x7x7
        embedding = out.view(-1, 392)
        embedding_out = F.relu(self.fc1(embedding))
        embedding = self.fc2(embedding_out)
        if (self.vae == 1):
            out = out.view(-1, 392)
            mu = self.mu(out)
            sigma = torch.exp(self.sigma(out))
            out = mu + sigma * self.N.sample(mu.shape)
            out = self.fc3(out)
            out = out.view(-1,8,7,7)
            self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()


        out = F.relu(self.t_conv1(out))
        out = F.relu(self.t_conv2(out))
        if(self.vae ==1 ):
            loss = ((x - out) ** 2).sum() + self.kl

        else:
            loss = F.mse_loss(out, x) + F.cross_entropy(embedding, y)
        self.log('valid_loss', loss)
        return loss


# model
from torchsummary import summary
model = autoencoder(vae=1)
# # summary(model, (1, 28, 28))
trainer = pl.Trainer(gpus=1,max_epochs = 1)
trainer.fit(model, train_loader, test_loader)
# torch.save(model.state_dict(), "./weights/initial/1.pt")




# #infer
#
# model.load_state_dict(torch.load("./weights/initial/1.pt"))
#

samples = 8
fig, ax = plt.subplots(samples,2 ,sharex='col', sharey='row')
# fig, ax1 = plt.subplots(4, sharex='col', sharey='row')

for i, batch in enumerate(train_loader):

    x,y = batch
    out,embed = model(x[:samples,:])
    it = 0

    for k in range(samples):
        ax[k][1].imshow(out[it,0,:].detach().numpy())

        ax[k][0].imshow(x[it,0,:])
        embe = embed[it,:].detach().numpy().reshape((4,4))
        # ax.set_title(str(y[k].item()))
        # ax[k][2].imshow(embe)
        # ax[i][k].title(y[it].item())
        it+=1
    plt.show()
    break


# TODO: Try Variational Auto Encoders
#       Efficient Net
