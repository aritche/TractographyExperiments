"""
Script for training a model
"""
import numpy as np
import os
import sys

#from models.cst_left_3d import CustomDataset, CustomModel, CustomLoss
from models.smaller_data import CustomDataset, CustomModel, CustomLoss
#from models.single_tract import CustomDataset, CustomModel, CustomLoss

from resources.vis import VisdomLinePlotter

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary


"""
Hyperparameters
"""
EPOCHS = 500
BATCH_SIZE = 16
LR = 10e-5
VALID_SPLIT = 0.15
np.random.seed(66)
torch.manual_seed(66)


"""
Result visualisation
"""
print('Initialising visualisation...')
plotter = VisdomLinePlotter(env_name='HairNet Training Experiment')

"""
Load the data
"""
print('Loading data...')
TOMs_path =        '../data/256_25_CST_left/preprocessed/TOMs'
beginnings_path =  '../data/256_25_CST_left/preprocessed/beginnings_masks'
endings_path =     '../data/256_25_CST_left/preprocessed/endings_masks'
tractograms_path = '../data/256_25_CST_left/not_preprocessed/tractograms'
if len(sys.argv) == 7:
    means = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
    sdevs = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    dataset = CustomDataset(TOMs_path, beginnings_path, endings_path, tractograms_path, means = means, sdevs = sdevs)
else:
    dataset = CustomDataset(TOMs_path, beginnings_path, endings_path, tractograms_path)

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)

#summary(model, (3, 256, 256))
summary(model, (3, 144, 144, 144))

# Create the results directory
model_name = input('** Name for the model (no spaces) [enter "dump" for no results]:')
if model_name != "dump":
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
        print(train_step)
        #print("Training epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, train_step, len(trainloader)))
        #print('Sending to GPU...')
        inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
        optimizer.zero_grad()
        #print('Stepping forward...')
        output = model.forward(inputs)

        #print('Computing loss...')
        loss = CustomLoss(output, labels)
        loss_item = loss.item()
        
        #print('Stepping backward...')
        loss.backward()

        #print('Optimising...')
        optimizer.step()
     
        #print('Updating loss...')
        # Since loss is mean over the batch, recover total loss across all items in batch
        batch_total_loss = loss_item * inputs.size(0)
        train_loss  += batch_total_loss
        train_step  += 1
        train_items += inputs.size(0)

        #print('Loading data...')

    plotter.plot('loss per item', 'train', 'Results', epoch, train_loss/train_items)

    valid_loss = 0.0
    valid_step = 0
    valid_items = 0
    model.eval()
    for inputs, labels in validloader:
        #print("Validation epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, valid_step, len(validloader)))
        inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
        output = model.forward(inputs)

        loss = CustomLoss(output, labels)
     
        valid_loss += loss.item()
        valid_step += 1
        valid_items += len(inputs)

    scheduler.step(valid_loss)
    plotter.plot('loss per item', 'validation', 'Results', epoch, valid_loss/valid_items)

    print("Epoch %d/%d:\t%.5f\t%.5f" % (epoch, EPOCHS, train_loss/train_items, valid_loss/valid_items))

    if epoch % 10 == 0:
        print('Saving intermediate model...')
        torch.save(model, './results/' + model_name + '/epoch_' + str(epoch) + '.pth')

    # Always save the most recent epoch
    print('Saving current model...')
    torch.save(model, './results/' + model_name + '/epoch_current.pth')

torch.save(model, './results/' + model_name + '/final_model_epoch_' + str(epoch) +'.pth')
