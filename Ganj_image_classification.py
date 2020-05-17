

import math
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Our libraries
# from train import train_model
# from model_utils import *
# from predict_utils import *
# from vis_utils import *

# some initial setup
np.set_printoptions(precision=2)
use_gpu = torch.cuda.is_available()
np.random.seed(1234)


DATA_DIR = 'C:/Python_machineLearning/dog vs cat/dataset/'
sz = 224
batch_size = 16

trn_dir = f'{DATA_DIR}training_set'
val_dir = f'{DATA_DIR}test_set'

print(os.listdir(DATA_DIR))
print(os.listdir(trn_dir))

trn_fnames = glob.glob(f'{trn_dir}/*/*.jpg')
trn_fnames[:5]


img = plt.imread(trn_fnames[3])
# plt.imshow(img)

################# Dataloader
train_ds = datasets.ImageFolder(trn_dir)

# print(train_ds.imgs)

################### Transformations
tfms = transforms.Compose([
    transforms.Resize((sz, sz)),  # PIL Image
    transforms.ToTensor(),        # Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(trn_dir, transform=tfms)
valid_ds = datasets.ImageFolder(val_dir, transform=tfms)

print(len(train_ds), len(valid_ds))

################### data loading from Train and validation dataset
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, 
                                       shuffle=True)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, 
                                       shuffle=True)

#inputs, targets = next(iter(train_dl))
# out = torchvision.utils.make_grid(inputs, padding=3)
# plt.figure(figsize=(16, 12))

################### define class of neural network

class Ganji_ImageClassifierCNN(nn.Module):
    
    def __init__(self):
        super(Ganji_ImageClassifierCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Linear(56 * 56 * 32, 2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)            # (bs, C, H,  W)
        out = out.view(out.size(0), -1)  # (bs, C * H, W)
        out = self.fc(out)
        return out

############# initiate model of NN

model = Ganji_ImageClassifierCNN()

# transfer model to GPU
if use_gpu:
    model = model.cuda()

########### Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()

############# Train Neural network
num_epochs = 3
losses = []
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dl):
        # inputs = to_var(inputs)
        # targets = to_var(targets)
        
        # forwad pass
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # loss
        loss = criterion(outputs, targets)
        losses += [loss.data]
        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()
        
        # report
        if (i + 1) % 50 == 0:
            print('Epoch [%2d/%2d], Step [%3d/%3d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_ds) // batch_size, loss.data))




