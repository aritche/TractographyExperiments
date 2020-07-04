import torch
import dipy
from dipy.io.image import load_nifti_data, load_nifti
import numpy as np
from dipy.io.streamline import load_trk
import nibabel as nib
from nibabel import trackvis
from dipy.tracking import utils

def load_TOM(fn):
    data = load_nifti_data(fn)
    return data

def crop_and_error(tom):
    cropped = tom[:,15:-14,:]

    diff = np.abs(np.sum(tom) - np.sum(cropped))
    rel = 100 * diff / np.sum(tom)

    return [cropped, diff, rel]


def load_streamlines(fn):
    tract = load_trk(fn, 'same', bbox_valid_check=False)
    streamlines = tract.streamlines
    return streamlines

def load_streamlines_v2(fn, ref):
    streams, header = trackvis.read(fn)

    data, ref_affine = load_nifti(ref)

    transformed = []
    for sl in streams:
        result = utils.apply_affine(aff=ref_affine, pts=sl[0])
        transformed.append(result)

    transformed = np.array(transformed)
    original = np.array([sl[0] for sl in streams])

    return [original, transformed]

def load_streamlines_v3(fn, ref):
    sl_file = nib.streamlines.load(fn)
    streamlines = sl_file.streamlines

    return streamlines

fn = '../../TractSeg_Replication/running_tractseg/tractseg_output_672756/TOM/CST_left.nii.gz'
sl = '../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms/672756/tracts/CA.trk'
ref_file = '../../DATASETS/HCP_100_SUBJECTS/672756/T1w/Diffusion/nodif_brain_mask.nii.gz'

sls = load_streamlines(sl)
[original, transformed] = load_streamlines_v2(sl, ref_file)
sls3 = load_streamlines_v3(sl, ref_file)

for i in range(len(sls)):
    for point in range(len(sls[i])):
        print(sls[i][point], original[i][point], transformed[i][point], sls3[i][point])

#sls = load_streamlines(sl)
#tom = load_TOM(fn)
#cropped, diff, rel = crop_and_error(tom)
