"""
This file generates pre-processed versions of TOMs, tract masks, and endings masks
It achieves this by finding a bounding box for the tract using the tract mask,
then cropping all files to this bounding box, then padding/scaling to achieve a size
of 144 x 144 x 144 x N
"""

from tractseg.libs import data_utils # assumes tractseg is installed
# ALTERNATIVELY, you could import it directly if you download the file from TractSeg > tractseg > libs > data_utils.py
# In this case, you would also need TractSeg > tractseg > libs > img_utils.py located in the same directory
# import data_utils

import nibabel as nib
import numpy as np
from glob import glob
import os


#fn = '../../DATASETS/HCP_100_SUBJECTS/100307/T1w/Diffusion/nodif_brain_mask.nii.gz' # nodif_brain_mask
base_dir = '../../DATASETS/TRACTSEG_105_SUBJECTS'
subjects = [subject.split('/')[-1] for subject in glob(base_dir + '/hcp_brain_masks/*')]

for subject in subjects:
    # Compute the bounding boxes for each tract
    bbox_dir = base_dir + '/generated_tract_masks/' + subject
    print(subject)

    # Base directories for output
    b1 = base_dir + '/preprocessed/generated_tract_masks/' + subject
    b2 = base_dir + '/preprocessed/generated_endings_masks/' + subject
    b3 = base_dir + '/preprocessed/generated_toms/' + subject
    for base in [b1, b2, b3]:
        if os.path.exists(base) == False:
            os.mkdir(base) # make the output subject directory

    for tract_fn in glob(bbox_dir + '/*'):
        # Load the mask file and compute bounding box
        mask = nib.load(tract_fn).get_data()
        if np.sum(mask) != 0:
            bbox = data_utils.get_bbox_from_mask(np.nan_to_num(mask), 0)
        else:
            bbox = [[0,mask.shape[0]], [0,mask.shape[1]], [0,mask.shape[2]]]

        tract_name = tract_fn.split('/')[-1].split('.')[0]

        # Input directories
        f1 = base_dir + '/generated_tract_masks/' + subject + '/' +  tract_name + '.trk.nii.gz'
        f2 = base_dir + '/generated_endings_masks/' + subject + '/' +  tract_name + '_beginnings.nii.gz'
        f3 = base_dir + '/generated_endings_masks/' + subject + '/' +  tract_name + '_endings.nii.gz'
        f4 = base_dir + '/generated_toms/' + subject + '/' +  tract_name + '_DIRECTIONS.nii.gz'
        
        # Output directories
        o1 = base_dir + '/preprocessed/generated_tract_masks/' + subject + '/' +  tract_name + '.trk.nii.gz'
        o2 = base_dir + '/preprocessed/generated_endings_masks/' + subject + '/' +  tract_name + '_beginnings.nii.gz'
        o3 = base_dir + '/preprocessed/generated_endings_masks/' + subject + '/' +  tract_name + '_endings.nii.gz'
        o4 = base_dir + '/preprocessed/generated_toms/' + subject + '/' +  tract_name + '_DIRECTIONS.nii.gz'


        fns = [f1, f2, f3, f4]
        outs = [o1, o2, o3, o4]

        i = 0
        for fn in fns:
            # Load the file you are interested in preprocessing
            if os.path.exists(fn):
                img = nib.load(fn)
                affine = img.affine
                data = img.get_data()
                
                # Adjust the data to have 4 dimensions, since the cropping method will iterate over this fourth dim
                if len(data.shape) == 3:
                    data = data[..., None]

                # Crop and pad to create a square volume of size 144 x 144 x 144
                data, _, _, original_shape = data_utils.crop_to_nonzero(np.nan_to_num(data), bbox=bbox)
                data, transform_applied = data_utils.pad_and_scale_img_to_square_img(data)

                # Save the result
                out_dir = outs[i]
                nib.save(nib.Nifti1Image(data, affine), out_dir)
                 
            i += 1
"""
# Load the nodif_brain_mask and get the bounding box of the brain region
mask = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/hcp_brain_masks/672756/nodif_brain_mask.nii.gz').get_data()
bbox = data_utils.get_bbox_from_mask(np.nan_to_num(mask), 0)
print(bbox)

# Load the file you are interested in preprocessing
#img = nib.load('../../DATASETS/tractseg_output_672756/TOM/CST_left.nii.gz')
#img = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/generated_toms/672756/CST_left_DIRECTIONS.nii.gz')
#img = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/generated_tract_masks/672756/CST_left.trk.nii.gz')
affine = img.affine
data = img.get_data()

# Adjust the data to have 4 dimensions, since the cropping method will iterate over this fourth dim
if len(data.shape) == 3:
    data = data[..., None]

# Crop and pad to create a square volume of size 144 x 144 x 144
data, _, _, original_shape = data_utils.crop_to_nonzero(np.nan_to_num(data), bbox=bbox)
data, transform_applied = data_utils.pad_and_scale_img_to_square_img(data)

# Save the result
nib.save(nib.Nifti1Image(data, affine), "result.nii.gz")

# Load the result
result = nib.load("result.nii.gz")

# Revert to the original shape
original = data_utils.cut_and_scale_img_back_to_original_img(data, transform_applied)
# NOTE: The following line takes parameter 'nr_of_classes'. This refers to the number of tracts (e.g. 72).
# I believe it uses this param for cases where the input dim and output dim for the deep network don't 
# match. E.g. input is 145 x 174 x 145 x 1 --> squared to 144 x 144 x 144 x 1 --> output of network 144 x 144 x 144 x 72
#             so now this function would convert it to 145 x 174 x 145 x 72
original = data_utils.add_original_zero_padding_again(original, bbox, original_shape, nr_of_classes=1)
print(original.shape)
nib.save(nib.Nifti1Image(original, affine), "original.nii.gz")
"""
