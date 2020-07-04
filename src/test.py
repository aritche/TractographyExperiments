"""
Script for testing a model
"""

import torch
import numpy as np
from models.single_tract import CustomDataset
from dipy.io.streamline import load_trk, save_trk


def save_as_trk(output, fn):
    output = output[0].cpu().detach().numpy()

    output = np.reshape(output, (1024,100,3))
    #output = output[0][0].cpu().numpy()
    #print(output)

    ref_trk = load_trk('../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms/672756/tracts/CST_left.trk', 'same', bbox_valid_check=False)
    sls = []
    for item in output:
        sls.append(item)
    ref_trk.streamlines = sls

    save_trk(ref_trk, fn + '.trk', bbox_valid_check=False)
        



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('final_model.pth')
model.to(device)
dataset = CustomDataset('../data/test/CST_TOMs', '../data/test/CST_endings_masks', '../data/CST_tractograms')
testloader = torch.utils.data.DataLoader(dataset, batch_size=1)

torch.cuda.empty_cache()
i = 0
for inputs, labels in testloader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model.forward(inputs)
    save_as_trk(output, str(i))
    i += 1
    
