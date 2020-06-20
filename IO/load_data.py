import torch
import dipy
from dipy.io.image import load_nifti_data
import numpy as np

def load_TOM(fn):
    data = load_nifti_data(fn)
    return data

def crop_and_error(tom):
    cropped = tom[:,15:-14,:]

    diff = np.abs(np.sum(tom) - np.sum(cropped))
    rel = 100 * diff / np.sum(tom)

    return [cropped, diff, rel]


fn = '../../TractSeg_Replication/running_tractseg/tractseg_output_672756/TOM/CST_left.nii.gz'
tom = load_TOM(fn)
cropped, diff, rel = crop_and_error(tom)
