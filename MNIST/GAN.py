import os
import time
import glob
import random
import imageio
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



# Custom LambdaLayer
class lambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(lambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(128, 392),
            lambdaLayer(lambda x: x-0.5),
            nn.CELU(0.2),
            nn.Linear(392, 784),
            lambdaLayer(lambda x: x-0.5),
            nn.CELU(0.2),
            nn.Linear(784, 784),
            lambdaLayer(lambda x: x-0.5),
            nn.CELU(1)
        )

    def forward(self, input):
        return self.main(input)


def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    for index, image in enumerate(images):
        plt.axis('off')
        plt.subplot(sqrtn, sqrtn, index+1)
        plt.imshow(image.reshape(28, 28))
        

def discriminator_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)


def generator_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)


def training_phase(gif_image = True):
    
    filenames = []
    for epoch in range(0, epochs):
    
        for ep, data in enumerate(train_loader):
            ep += 1
            
            # Discriminator
            real_inputs = data[0].to(device)
            test = 255 * (0.5 * real_inputs[0] + 0.5)
            real_inputs = real_inputs.view(-1, 784)
            real_outputs = Discriminator(real_inputs)
            real_label = torch.ones(real_inputs.shape[0], 1).to(device)
    
            noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
            noise = noise.to(device)
            fake_inputs = Generator(noise)
            fake_outputs = Discriminator(fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)
    
            outputs = torch.cat((real_outputs, fake_outputs), 0)
            targets = torch.cat((real_label, fake_label), 0)
    
            discriminator_optimizer.zero_grad()
            discriminator_loss = discriminator_loss_function(outputs, targets)
            discriminator_loss.backward()
            discriminator_optimizer.step()
    
            # Generator
            noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
            noise = noise.to(device)
    
            fake_inputs = Generator(noise)
            fake_outputs = Discriminator(fake_inputs)
    
            generator_loss = generator_loss_function(fake_outputs)
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()
    
            if ep % batch_size == 0 or ep == len(train_loader):
                print('Epochs:[{}/{}] ({:.0f}%) Discriminator loss: {:.3f} Generator loss: {:.3f}'
                      .format(epoch, epochs, (ep/len(train_loader)*100), discriminator_loss.item(), generator_loss.item()))
    
        if gif_image:
            imgs_numpy = (fake_inputs.data.cpu().numpy()+1.0)/2.0
            show_images(imgs_numpy[:16])
            plt.axis('off')
            plt.savefig('img{}.png'.format(epoch), bbox_inches='tight')
            plt.close()
            filenames.append('img{}.png'.format(epoch))
           
    print('')
    print('Training Finished {:.0f} seconds'.format(time.time()-start_time))
    
    if gif_image:
        with imageio.get_writer('gif_GAN.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        for filename in set(filenames):
            os.remove(filename)




# Hyperparameters
epochs = 30
batch_size = 64
betas = (0.5, 0.999)
learning_rate = 0.0002
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])



# Main Code    
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)
    
start_time = time.time()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.ioff()


Generator = generator().to(device)
Discriminator = discriminator().to(device)

generator_optimizer = optim.Adam(Generator.parameters(), lr=learning_rate, betas=betas)
discriminator_optimizer = optim.Adam(Discriminator.parameters(), lr=learning_rate, betas=betas)


# Load data
train_set = datasets.MNIST('mnist/', train=True, download=True, transform=transform)
test_set = datasets.MNIST('mnist/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Training
images = training_phase(True)

# Generator
noise = (torch.rand(16, 128)-0.5) / 0.5
noise = noise.to(device)

fake_image = Generator(noise)
imgs_numpy = (fake_image.data.cpu().numpy()+1.0)/2.0
show_images(imgs_numpy)
plt.show()