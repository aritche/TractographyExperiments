"""
Script for training a model
"""
import numpy as np
import os
import sys
from dipy.io.image import load_nifti 

import cv2

#from models.cst_left_3d import CustomDataset, CustomModel, CustomLoss
from models.final_hairnet import CustomDataset, CustomModel, CustomLoss, OutputToStreamlines
#from models.seeds_as_input import CustomDataset, CustomModel, CustomLoss
#from models.single_tract import CustomDataset, CustomModel, CustomLoss

from resources.vis import VisdomLinePlotter
import visdom
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary

import time


"""
Hyperparameters
"""
EPOCHS = 500
BATCH_SIZE = 4
LR = 10e-4
VALID_SPLIT = 0.15
np.random.seed(66)
torch.manual_seed(66)


"""
Result visualisation
"""
print('Initialising visualisation...')
plotter = VisdomLinePlotter(env_name='HairNet Training Experiment')
image_plotter = visdom.Visdom()

"""
Load the data
"""
print('Loading data...')

TOMs_path =        '../data/final_hairnet_dataset/preprocessed/TOMs'
seeds_path =       '../data/final_hairnet_dataset/preprocessed/seeds'
beginnings_path =  '../data/final_hairnet_dataset/preprocessed/seeds' # NOTE this difference
endings_path =     '../data/final_hairnet_dataset/preprocessed/ends'  # NOTE this difference
tractograms_path = '../data/final_hairnet_dataset/not_preprocessed/tractograms'

#TOMs_path =        '../data/concat_cst_left/preprocessed/TOMs'
#seeds_path =       '../data/concat_cst_left/preprocessed/seeds'
#beginnings_path =  '../data/concat_cst_left/preprocessed/beginnings_masks'
#endings_path =     '../data/concat_cst_left/preprocessed/endings_masks'
#tractograms_path = '../data/concat_cst_left/not_preprocessed/tractograms'

#TOMs_path =        '../data/concat_final_hairnet/preprocessed/TOMs'
#seeds_path =       '../data/concat_final_hairnet/preprocessed/seeds'
#beginnings_path =  '../data/concat_final_hairnet/preprocessed/beginnings_masks'
#endings_path =     '../data/concat_final_hairnet/preprocessed/endings_masks'
#tractograms_path = '../data/concat_final_hairnet/not_preprocessed/tractograms'

#TOMs_path =        '../data/seeds_1024_32_CST_left/preprocessed/TOMs'
#seeds_path =       '../data/seeds_1024_32_CST_left/preprocessed/seeds'
#beginnings_path =  '../data/seeds_1024_32_CST_left/preprocessed/beginnings_masks'
#endings_path =     '../data/seeds_1024_32_CST_left/preprocessed/endings_masks'
#tractograms_path = '../data/seeds_1024_32_CST_left/not_preprocessed/tractograms'

#TOMs_path =        '../data/1024_40_CST_left_fixed/preprocessed/TOMs'
#beginnings_path =  '../data/1024_40_CST_left_fixed/preprocessed/beginnings_masks'
#endings_path =     '../data/1024_40_CST_left_fixed/preprocessed/endings_masks'
#tractograms_path = '../data/1024_40_CST_left_fixed/not_preprocessed/tractograms'

#TOMs_path =        '../data/1024_40_CST_left_rectified/preprocessed/TOMs'
#beginnings_path =  '../data/1024_40_CST_left_rectified/preprocessed/beginnings_masks'
#endings_path =     '../data/1024_40_CST_left_rectified/preprocessed/endings_masks'
#tractograms_path = '../data/1024_40_CST_left_rectified/not_preprocessed/tractograms'

#TOMs_path =        '../data/64_40_CST_left_rectified_1000/preprocessed/TOMs'
#beginnings_path =  '../data/64_40_CST_left_rectified_1000/preprocessed/beginnings_masks'
#endings_path =     '../data/64_40_CST_left_rectified_1000/preprocessed/endings_masks'
#tractograms_path = '../data/64_40_CST_left_rectified_1000/not_preprocessed/tractograms'

#TOMs_path =        '../data/64_40_CST_left_rectified/preprocessed/TOMs'
#beginnings_path =  '../data/64_40_CST_left_rectified/preprocessed/beginnings_masks'
#endings_path =     '../data/64_40_CST_left_rectified/preprocessed/endings_masks'
#tractograms_path = '../data/64_40_CST_left_rectified/not_preprocessed/tractograms'

#TOMs_path =        '../data/64_40_CST_left/preprocessed/TOMs'
#beginnings_path =  '../data/64_40_CST_left/preprocessed/beginnings_masks'
#endings_path =     '../data/64_40_CST_left/preprocessed/endings_masks'
#tractograms_path = '../data/64_40_CST_left/not_preprocessed/tractograms'

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

def fast_vis(inputs, outputs, num_sl):
    # Get first item of batch
    seeds, output = inputs[1][0], outputs[0]

    # Get streamlines and seeds from tensors
    streamlines = OutputToStreamlines(output)
    seeds = seeds.permute(1, 0).cpu().detach().numpy()

    # Reconstruct streamlines
    for i in range(len(streamlines)):
        streamlines[i] += seeds[i]

    # Sample streamlines
    streamlines = streamlines[np.random.choice(streamlines.shape[0],num_sl, replace=False)]

    # Plot the result
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    for i in range(num_sl):
        ax.plot(streamlines[i,:,0], streamlines[i,:,1], streamlines[i,:,2])

    # Render the result
    fig.tight_layout()
    fig.canvas.draw()

    # Convert image of plot to a numpy array
    im = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im = im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Tranpose, as visdom expect channels first
    im = np.transpose(im, (2,0,1))

    return im

_, affine = load_nifti(TOMs_path + '/599469_0_CST_left.nii.gz')
inverse_affine = np.linalg.inv(affine)
if len(sys.argv) == 7:
    means = [float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])]
    sdevs = [float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])]
    dataset = CustomDataset(TOMs_path, tractograms_path, beginnings_path, endings_path, inverse_affine, means = means, sdevs = sdevs)
else:
    dataset = CustomDataset(TOMs_path, tractograms_path, beginnings_path, endings_path, inverse_affine)

# Split into training/validation (https://stackoverflow.com/a/50544887)
indices = list(range(len(dataset)))
split = int(np.floor(VALID_SPLIT * len(dataset)))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

trainloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE, drop_last=True)
validloader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=BATCH_SIZE, drop_last=True)

"""
Train the model
"""
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

to_load = input("Do you want to load a model and continue training? [Y/N]")
if to_load == 'Y' or to_load == 'y':
    model_name = input("Model name:")
    epoch_number = input("Epoch:")
    model_fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'
    while not os.path.exists('./results/' + model_name):
        print("Model does not exist. Please try again.")
        model_name = input("Model name:")
        epoch_number = input("Epoch:")
        model_fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'
    model = torch.load(model_fn)
else:
    # Create the model
    model = CustomModel()

    # Create the results directory
    model_name = input('** Name for the model (no spaces) [enter "dump" for no results]:')
    if model_name != "dump":
        while os.path.exists('./results/' + model_name):
            model_name = input('Model already exists. Please enter another name:')
        os.mkdir('./results/' + model_name) 

# Send to device
model.to(device)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)

# Print a summary
#summary(model, [(5, 144, 144, 144), (3,1024)])
input('Press any key to begin training...')

#scaler = torch.cuda.amp.GradScaler()
print("Training...")

start_epoch = 1
if to_load == 'Y' or to_load == 'y':
    if epoch_number == 'current':
        start_epoch = int(input("What epoch does 'current' correspond to?"))
    else:
        start_epoch = int(epoch_number)
    start_epoch += 1

for epoch in range(start_epoch,EPOCHS):

    train_loss = 0.0
    train_step = 0
    train_items = 0
    model.train()
    t0 = time.time()
    for inputs, labels in trainloader:
        print("Data loading time %.3f" % (time.time() - t0))

        print(train_step)

        ##################################
        t0 = time.time()

        #print("Training epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, train_step, len(trainloader)))
        #print('Sending to GPU...')

        # Send inputs to device
        if type(inputs) is list:
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device)
        else:
            inputs = inputs.to(device)

        # Send labels to device
        if type(labels) is list:
            for i in range(len(labels)):
                labels[i] = labels[i].to(device)
        else:
            labels = labels.to(device)

        optimizer.zero_grad()

        #print('Stepping forward...')
        if type(inputs) is list:
            output = model.forward(*inputs)
        else:
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
        if type(inputs) is list:
            num_items = inputs[0].size(0)
        else:
            num_items = inputs.size(0)

        batch_total_loss = loss_item * num_items
        train_loss  += batch_total_loss
        train_step  += 1
        train_items += num_items


        print("Training time %.3f" % (time.time() - t0))
        print('^^^^^^^^^')
        ############################

        #print('Loading data...')
        t0 = time.time()
        #break

    plotter.plot('loss per item', 'train', 'Results', epoch, train_loss/train_items)

    valid_loss = 0.0
    valid_step = 0
    valid_items = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            #print("Validation epoch %d/%d (step %d/%d)" % (epoch, EPOCHS, valid_step, len(validloader)))
            # Send inputs to device
            if type(inputs) is list:
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device)
            else:
                inputs = inputs.to(device)

            # Send labels to device 
            if type(labels) is list:
                for i in range(len(labels)):
                    labels[i] = labels[i].to(device)
            else:
                labels = labels.to(device)

            if type(inputs) is list:
                output = model.forward(*inputs)
            else:
                output = model.forward(inputs)

            loss = CustomLoss(output, labels)
            loss_item = loss.item()
         
            if type(inputs) is list:
                num_items = inputs[0].size(0)
            else:
                num_items = inputs.size(0)

            valid_loss += loss_item * num_items
            valid_step += 1
            valid_items += num_items
            #break
            
    plotter.plot('loss per item', 'validation', 'Results', epoch, valid_loss/valid_items)

    # Plot the results
    label_im = fast_vis(inputs, labels, 20)
    gen_im = fast_vis(inputs, output, 20)
    image_plotter.images([label_im, gen_im], opts=dict(caption='Epoch ' + str(epoch)))

    scheduler.step(valid_loss)

    print("Epoch %d/%d:\t%.5f\t%.5f" % (epoch, EPOCHS, train_loss/train_items, valid_loss/valid_items))

    if epoch % 10 == 0:
        print('Saving intermediate model...')
        torch.save(model, './results/' + model_name + '/epoch_' + str(epoch) + '.pth')

    # Always save the most recent epoch
    print('Saving current model...')
    torch.save(model, './results/' + model_name + '/epoch_current.pth')

torch.save(model, './results/' + model_name + '/final_model_epoch_' + str(epoch) +'.pth')
