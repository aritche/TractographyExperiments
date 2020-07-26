"""
Script for training a model
"""
import numpy as np
import os
import sys

#from models.cst_left_3d import CustomDataset, CustomModel, CustomLoss
from models.custom_dataset_small import CustomDataset, CustomModel, CustomLoss
#from models.single_tract import CustomDataset, CustomModel, CustomLoss

from resources.vis import VisdomLinePlotter

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

import time


"""
Hyperparameters
"""
EPOCHS = 500
BATCH_SIZE = 16
LR = 10e-4
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

TOMs_path =        '../data/64_40_CST_left/preprocessed/TOMs'
beginnings_path =  '../data/64_40_CST_left/preprocessed/beginnings_masks'
endings_path =     '../data/64_40_CST_left/preprocessed/endings_masks'
tractograms_path = '../data/64_40_CST_left/not_preprocessed/tractograms'

#TOMs_path =        '../data/custom_dataset_105/preprocessed/TOMs'
#beginnings_path =  '../data/custom_dataset_105/preprocessed/beginnings_masks'
#endings_path =     '../data/custom_dataset_105/preprocessed/endings_masks'
#tractograms_path = '../data/custom_dataset_105/not_preprocessed/tractograms'

#TOMs_path =        '../data/custom_dataset_1000/preprocessed/TOMs'
#beginnings_path =  '../data/custom_dataset_1000/preprocessed/beginnings_masks'
#endings_path =     '../data/custom_dataset_1000/preprocessed/endings_masks'
#tractograms_path = '../data/custom_dataset_1000/not_preprocessed/tractograms'

#TOMs_path =        '../data/custom_dataset/preprocessed/TOMs'
#beginnings_path =  '../data/custom_dataset/preprocessed/beginnings_masks'
#endings_path =     '../data/custom_dataset/preprocessed/endings_masks'
#tractograms_path = '../data/custom_dataset/not_preprocessed/tractograms'

#TOMs_path =        '../data/QB_4_15_CST_left_based_on_256_25/preprocessed/TOMs'
#beginnings_path =  '../data/QB_4_15_CST_left_based_on_256_25/preprocessed/beginnings_masks'
#endings_path =     '../data/QB_4_15_CST_left_based_on_256_25/preprocessed/endings_masks'
#tractograms_path = '../data/QB_4_15_CST_left_based_on_256_25/not_preprocessed/tractograms'

#TOMs_path =        '../data/QB_5_15_CST_left/preprocessed/TOMs'
#beginnings_path =  '../data/QB_5_15_CST_left/preprocessed/beginnings_masks'
#endings_path =     '../data/QB_5_15_CST_left/preprocessed/endings_masks'
#tractograms_path = '../data/QB_5_15_CST_left/not_preprocessed/tractograms'

#TOMs_path =        '../data/256_25_CST_left/preprocessed/TOMs'
#beginnings_path =  '../data/256_25_CST_left/preprocessed/beginnings_masks'
#endings_path =     '../data/256_25_CST_left/preprocessed/endings_masks'
#tractograms_path = '../data/256_25_CST_left/not_preprocessed/tractograms'

#TOMs_path =        '../data/PRE_SAMPLED/preprocessed/TOMs'
#beginnings_path =  '../data/PRE_SAMPLED/preprocessed/beginnings_masks'
#endings_path =     '../data/PRE_SAMPLED/preprocessed/endings_masks'
#tractograms_path = '../data/PRE_SAMPLED/tractograms'
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


#scaler = torch.cuda.amp.GradScaler()
print("Training...")
for epoch in range(EPOCHS):

    train_loss = 0.0
    train_step = 0
    train_items = 0
    model.train()
    #t0 = time.time()
    for inputs, labels in trainloader:
        #print(time.time() - t0)

        print(train_step)

        ##################################
        #t0 = time.time()

        #print("Training epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, train_step, len(trainloader)))
        #print('Sending to GPU...')
        inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
        #inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        #print('Stepping forward...')
        output = model.forward(inputs)

        #print('Computing loss...')
        loss = CustomLoss(output, labels)
        loss_item = loss.item()
        
        #print('Stepping backward...')
        #scaler.scale(loss).backward()
        loss.backward()

        #print('Optimising...')
        #scaler.step(optimizer)#.step()
        optimizer.step()

        #scaler.update()
     
        #print('Updating loss...')
        # Since loss is mean over the batch, recover total loss across all items in batch
        batch_total_loss = loss_item * inputs.size(0)
        train_loss  += batch_total_loss
        train_step  += 1
        train_items += inputs.size(0)


        #print(time.time() - t0)
        #print('^^^^^^^^^')
        ############################

        #print('Loading data...')
        #t0 = time.time()

    
    plotter.plot('loss per item', 'train', 'Results', epoch, train_loss/train_items)

    valid_loss = 0.0
    valid_step = 0
    valid_items = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            #print("Validation epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, valid_step, len(validloader)))
            inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
            #inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)

            loss = CustomLoss(output, labels)
            loss_item = loss.item()
         
            valid_loss += loss_item * inputs.size(0)
            valid_step += 1
            valid_items += inputs.size(0)

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
