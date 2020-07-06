"""
Script for testing a model
"""

import torch
import numpy as np
from models.single_tract_3d import CustomDataset
from dipy.io.streamline import load_trk, save_trk
import matplotlib.pyplot as plt


def save_as_trk(output, fn):
    output = output[0].cpu().detach().numpy()

    output = np.reshape(output, (1024,100,3))

    ref_trk = load_trk('../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms/672756/tracts/CST_left.trk', 'same', bbox_valid_check=False)

    sls = []
    for item in output:
        sls.append(item)
    ref_trk.streamlines = sls
    save_trk(ref_trk, fn + '.trk', bbox_valid_check=False)

    plt.hist(output.ravel())
    plt.show()

    return output


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
dataset = CustomDataset(test_dir + '/CST_TOMs', test_dir + '/CST_endings_masks', test_dir + '/CST_tractograms')
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

torch.cuda.empty_cache()
i = 0
results = []
for inputs, labels in testloader:
    TEMP = labels[0].cpu().detach().numpy()   
    plt.hist(TEMP.ravel())
    plt.show()

    inputs, labels = inputs.to(device), labels.to(device)
    output = model.forward(inputs)

    result = save_as_trk(output, str(i))
    results.append(result)
    i += 1

print(np.sum(np.abs(results[0] - results[1])))
