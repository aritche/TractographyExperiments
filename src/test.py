"""
Script for testing a model
"""

import torch
import numpy as np
from models.relative_sorted import CustomDataset, OutputToStreamlines
from dipy.io.streamline import load_trk, save_trk

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2

from plotly import graph_objs as go
import plotly.express as px

import sys
import random

def save_as_trk(output, fn):
    output = OutputToStreamlines(output)

    ref_trk = load_trk('../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms/672756/tracts/CST_left.trk', 'same', bbox_valid_check=False)

    sls = []
    for item in output:
        sls.append(item)
    ref_trk.streamlines = sls
    save_trk(ref_trk, fn + '.trk', bbox_valid_check=False)

def plot_streamlines_plotly(output):
    seeds, positions = output

    # Conver to a list of streamlines
    streamlines = OutputToStreamlines(output)
    colors = []
    i = 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    coords = np.reshape(streamlines, (-1, 3))
    fig = px.line_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], color=colors)
    fig.show()

def plot_streamlines(output, n, shuffle=False):
    seeds, positions = output

    # Conver to a list of streamlines
    streamlines = OutputToStreamlines(output)

    # Plot the first 3 streamlines, one at a time
    x, y, z = [], [], []
    u, v, w = [], [], []

    streamlines_copy = streamlines.copy()
    if shuffle == True:
        random.shuffle(streamlines_copy)
    for sl in streamlines_copy[:n]:
        for i in range(len(sl)-1):
            us, vs, ws = sl[i+1] - sl[i]
            dist = (us**2 + vs**2 + ws**2)**(1/2)
            #us, vs, ws = us/dist, vs/dist, ws/dist
            u.append(us)
            v.append(vs)
            w.append(ws)
            x.append(sl[i][0])
            y.append(sl[i][1])
            z.append(sl[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, u, v, w)

    # Plot the seeds on the same figure
    seeds = seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    seeds = seeds.cpu().detach().numpy()
    seeds = np.reshape(seeds, (-1, 3))
    seeds_x = list(seeds[:,0])
    seeds_y = list(seeds[:,1])
    seeds_z = list(seeds[:,2])

    ax.scatter(seeds_x, seeds_y, seeds_z, c='r')
    plt.show()

def plot_vectors(outputs):
    for output in outputs:
        streamlines = OutputToStreamlines(output)
        x, y, z = [], [], []
        u, v, w = [], [], []
        for sl in streamlines:
            for i in range(len(sl)-1):
                us, vs, ws = sl[i+1] - sl[i]
                dist = (us**2 + vs**2 + ws**2)**(1/2)
                us, vs, ws = us/dist, vs/dist, ws/dist
                u.append(us)
                v.append(vs)
                w.append(ws)
                x.append(sl[i][0])
                y.append(sl[i][1])
                z.append(sl[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(x, y, z, u, v, w)
    plt.show()

def plot_seeds(output):
    seeds = output[0].permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    seeds = seeds.cpu().detach().numpy()
    seeds = np.reshape(seeds, (-1, 3))

    x = seeds[:,0]
    y = seeds[:,1]
    z = seeds[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(x), list(y), list(z))
    plt.show()

def plot_outputs(outputs):
    seeds, positions = outputs
    output = [seeds[0], positions[0]]

    # Conver to a list of streamlines
    output = OutputToStreamlines(output)

    # Get the list of coordinates
    coords = np.reshape(output, (-1, 3))

    # Extract x, y and z
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]

    # Plot histograms
    gs = gridspec.GridSpec(1,3)
    plt.figure()
    ax = plt.subplot(gs[0,0])
    plt.hist(x)
    ax = plt.subplot(gs[0,1])
    plt.hist(y)
    ax = plt.subplot(gs[0,2])
    plt.hist(z)
    plt.show()

    # Show output volume
    #for i in range(len(x)):
        #print(x[i], y[i], z[i])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(x), list(y), list(z))
    plt.show()
    
def plot_two(a, b):
    a = OutputToStreamlines(a[0])
    b = OutputToStreamlines(b[0])

    coordsA = np.reshape(a, (-1,3))
    coordsB = np.reshape(b, (-1,3))

    x_a = coordsA[:,0]
    y_a = coordsA[:,1]
    z_a = coordsA[:,2]

    x_b = coordsB[:,0]
    y_b = coordsB[:,1]
    z_b = coordsB[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(x_a), list(y_a), list(z_a))
    ax.scatter(list(x_b), list(y_b), list(z_b))

    plt.show()

def plot_output_volume(output, label):
    seeds, coords = output
    label_seeds, label_coords = label

    seeds = seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    seeds = seeds.cpu().detach().numpy()
    label_seeds = label_seeds.permute(1, 2, 0) # (3, 32, 32) -> (32, 32, 3)
    label_seeds = label_seeds.cpu().detach().numpy()

    im_x = seeds[:,:,0]
    im_y = seeds[:,:,1]
    im_z = seeds[:,:,2]
    im_x = (im_x - np.min(im_x)) / (np.max(im_x) - np.min(im_x))*255
    im_y = (im_y - np.min(im_y)) / (np.max(im_y) - np.min(im_y))*255
    im_z = (im_z - np.min(im_z)) / (np.max(im_z) - np.min(im_z))*255
    seeds_norm = np.stack([im_x, im_y, im_z], axis=2)
    cv2.namedWindow('seeds_norm', cv2.WINDOW_NORMAL)


    im_x = label_seeds[:,:,0]
    im_y = label_seeds[:,:,1]
    im_z = label_seeds[:,:,2]
    im_x = (im_x - np.min(im_x)) / (np.max(im_x) - np.min(im_x))*255
    im_y = (im_y - np.min(im_y)) / (np.max(im_y) - np.min(im_y))*255
    im_z = (im_z - np.min(im_z)) / (np.max(im_z) - np.min(im_z))*255
    label_seeds_norm = np.stack([im_x, im_y, im_z], axis=2)
    cv2.namedWindow('label_seeds_norm', cv2.WINDOW_NORMAL)


    cv2.imshow('seeds_norm', np.uint8(seeds_norm))
    cv2.imshow('label_seeds_norm', np.uint8(label_seeds_norm))
    cv2.waitKey(0)
    """
    cv2.imshow('seeds', np.uint8(seeds))
    cv2.imshow('label_seeds', np.uint8(label_seeds))

    plot_seeds(label)
    plot_seeds(output)
    cv2.waitKey(0)

    coords = coords.permute(1, 2, 0) # (120, 32, 32) -> (32, 32, 120)
    coords = coords.cpu().detach().numpy()
    coords = np.reshape(coords, (32, 32, 40, 3)) # convert to RGB volume
    label_coords = label_coords.permute(1, 2, 0) # (120, 32, 32) -> (32, 32, 120)
    label_coords = label_coords.cpu().detach().numpy()
    label_coords = np.reshape(label_coords, (32, 32, 40, 3)) # convert to RGB volume

    cv2.namedWindow('label outputs', cv2.WINDOW_NORMAL)
    cv2.namedWindow('output raw volume', cv2.WINDOW_NORMAL)
    for i in range(len(label_coords[2])):
        slice = label_coords[:,:,i,:]
        cv2.imshow('label outputs', np.uint8(slice))

        slice = coords[:,:,i,:]
        cv2.imshow('output raw volume', np.uint8(slice))

        cv2.waitKey(0)

    tractogram = OutputToStreamlines(output)
    label_tractogram = OutputToStreamlines(label)
    """


def plot_input_volume(inputs):
    for input in inputs:
        input = input.permute(1, 2, 3, 0) # (40*3, 32, 32) -> (32, 32, 40*3)
        input = input.cpu().detach().numpy()

        for i in range(len(input)):
            slice = input[i,:,:,:]
            #print(slice.shape)
            #slice = (slice - np.min(slice)) / (np.max(slice) - np.min(slice))
            #slice *= 255
            slice = np.uint8(slice)
            cv2.imshow('input', slice)
            cv2.waitKey(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dump = input("Pleas make sure you are importing the correct model into the test file...")

# Parameters
args = sys.argv
model_name = args[1]
epoch_number = args[2]
test_dir = '../data/' + args[3]
means = np.array([-0.000018, 0.000004, -0.000012])
sdevs = np.array([0.048257, 0.039410, 0.088247])

# Load model
print("Loading model...")
fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'
model = torch.load(fn)
model.to(device)

# Load dataset
dataset = CustomDataset(test_dir + '/CST_TOMs', test_dir + '/CST_beginnings_masks', test_dir + '/CST_endings_masks', test_dir + '/CST_tractograms', means=means, sdevs=sdevs) 
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

torch.cuda.empty_cache()

i = 0
for inputs, labels in testloader:
    # Pass to the model
    inputs, labels = inputs.to(device), [labels[0].to(device), labels[1].to(device)]
    outputs = model.forward(inputs)

    # Remove the batching to get the single item
    input = inputs[0]
    label = [labels[0][0], labels[1][0]]
    output = [outputs[0][0], outputs[1][0]]

    # Plot the generated volume
    #plot_output_volume(output, label)

    # Show the seeds
    #plot_seeds(label)
    #plot_seeds(output)
    #plot_streamlines(label, 30, shuffle=True)
    #plot_streamlines(output, 30, shuffle=True)
    plot_output_volume(output, label)

    #save_as_trk(output[0], str(i))
    i += 1
