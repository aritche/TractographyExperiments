"""
Script for training a model
"""
import numpy as np
from models.single_tract import CustomDataset, CustomModel

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary


"""
Hyperparameters
"""
EPOCHS = 500
BATCH_SIZE = 32
LR = 10e-4
VALID_SPLIT = 0.2
np.random.seed(66)

"""
Load the data
"""
dataset = CustomDataset('../data/CST_TOMs', '../data/CST_endings_masks', '../data/CST_tractograms')

# Split into training/validation (https://stackoverflow.com/a/50544887)
indices = list(range(len(dataset)))
split = int(np.floor(VALID_SPLIT * len(dataset)))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trainloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=BATCH_SIZE)

"""
Train the model
"""

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

model = CustomModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

summary(model, (3, 256, 256))

"""
for epoch in range(EPOCHS):
    step = 0
    for inputs, labels in trainloader:
        print("Training epoch %d/%d (step %d/%d)" % (EPOCHS, len(trainloader)))
        optimizer.zero_grad()
        output = model.forward(inputs)

        loss = CustomLoss(output, labels)
        loss.backward()
        optimizer.step()
     
        step += 1
"""
