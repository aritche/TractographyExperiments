"""
Script for testing a model
"""

import torch
import numpy as np
from models.cst_left_3d import CustomDataset, OutputToStreamlines
from dipy.io.streamline import load_trk, save_trk

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import cv2

def save_as_trk(output, fn):
    output = OutputToStreamlines(output)

    ref_trk = load_trk('../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms/672756/tracts/CST_left.trk', 'same', bbox_valid_check=False)

    sls = []
    for item in output:
        sls.append(item)
    ref_trk.streamlines = sls
    save_trk(ref_trk, fn + '.trk', bbox_valid_check=False)

def plot_outputs(outputs):
    for output in outputs:
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
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = input("Model name:")
epoch_number = input("Epoch number:")
test_dir = input("Test Directory:")
test_dir = "../data/" + test_dir

# Load model
fn = './results/' + model_name + '/epoch_' + epoch_number + '.pth'
model = torch.load(fn)
model.to(device)

# Load dataset
dataset = CustomDataset(test_dir + '/CST_TOMs', test_dir + '/CST_endings_masks', test_dir + '/CST_tractograms', means=np.array([-0.000011, -0.000014, 0.000008]), sdevs=np.array([0.059968, 0.047454, 0.099328]))
#dataset = CustomDataset('../data/CST_TOMs', '../data/CST_endings_masks', '../data/CST_tractograms_raw', means=np.array([-0.000011, -0.000014, 0.000008]), sdevs=np.array([0.059968, 0.047454, 0.099328]))
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

torch.cuda.empty_cache()
i = 0
for inputs, labels in testloader:
    # Show the target distribution
    print('Plotting ground truth...')
    print(labels.size())
    plot_outputs(labels)

    # Pass to the model
    inputs, labels = inputs.to(device), labels.to(device)
    output = model.forward(inputs)

    print('Plotting generated outputs...')
    plot_outputs(output)

    #save_as_trk([0], str(i))
    i += 1
