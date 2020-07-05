"""
Script for training a model
"""
import numpy as np
import os

from models.single_tract import CustomDataset, CustomModel, CustomLoss

from resources.vis import VisdomLinePlotter

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
Result visualisation
"""
print('Initialising visualisation...')
plotter = VisdomLinePlotter(env_name='HairNet Training Experiment')

"""
Load the data
"""
print('Loading data...')
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

# Create the results directory
model_name = input('** Name for the model (no spaces):')
while os.path.exists('./results/' + model_name):
    model_name = input('Model already exists. Please enter another name:')
os.mkdir('./results/' + model_name) 


print("Training...")
for epoch in range(EPOCHS):

    train_loss = 0.0
    train_step = 0
    train_items = 0
    model.train()
    for inputs, labels in trainloader:
        #print("Training epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, train_step, len(trainloader)))
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(inputs)

        loss = CustomLoss(output, labels)
        loss.backward()
        optimizer.step()
     
        train_loss  += loss.item()
        train_step  += 1
        train_items += len(inputs)

    plotter.plot('loss per item', 'train', 'Results', epoch, train_loss/train_items)

    valid_loss = 0.0
    valid_step = 0
    valid_items = 0
    model.eval()
    for inputs, labels in validloader:
        #print("Validation epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, valid_step, len(validloader)))
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)

        loss = CustomLoss(output, labels)
     
        valid_loss += loss.item()
        valid_step += 1
        valid_items += len(inputs)

    plotter.plot('loss per item', 'validation', 'Results', epoch, valid_loss/valid_items)

    print("Epoch %d/%d:\t%.5f\t%.5f" % (epoch, EPOCHS, train_loss, valid_loss))

    if epoch % 100 == 0:
        print('Saving intermediate model...')
        torch.save(model, './results/' + model_name + '/epoch_' + str(epoch) + '.pth')

torch.save(model, './results/' + model_name + '/final_model_epoch_' + str(epoch) +'.pth')
