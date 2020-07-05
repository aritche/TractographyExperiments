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
from dipy.tracking import utils
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from dipy.io.streamline import load_trk, save_trk
import numpy as np
from glob import glob
import os
import re


def preprocess_nifti_files():
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

def preprocess_trk_files():
    base_dir   = '../../DATASETS/TRACTSEG_105_SUBJECTS'
    input_dir  = '../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms'
    output_dir = '../../DATASETS/TRACTSEG_105_SUBJECTS/preprocessed/tractograms'
    subjects = [subject.split('/')[-1] for subject in glob(input_dir + '/*')]

    for subject in subjects:
        for fn in glob(input_dir + '/' + subject + '/tracts/*.trk'):
            tract_name = fn.split('/')[-1].split('.')[0]

            # Open the file
            # Method of opening trk files >= V1.2.0
            #trk_file = nib.streamlines.load(fn)
            trk_file = load_trk(fn, 'same', bbox_valid_check=False)
            print(fn)
            streamlines = trk_file.streamlines

            # Get the transformation applied to nifti file
            bbox_dir = base_dir + '/generated_tract_masks/' + subject
            img = nib.load(base_dir + '/generated_tract_masks/' + subject + '/' + tract_name + '.trk.nii.gz')
            affine = img.affine
            mask = img.get_data()
            if np.sum(mask) != 0:
                bbox = data_utils.get_bbox_from_mask(np.nan_to_num(mask), 0)
            else:
                bbox = [[0,mask.shape[0]], [0,mask.shape[1]], [0,mask.shape[2]]]

            # Adjust the data to have 4 dimensions, since the cropping method will iterate over this fourth dim
            if len(mask.shape) == 3:
                mask = mask[..., None]
            mask, _, _, _ = data_utils.crop_to_nonzero(np.nan_to_num(mask), bbox=bbox)
            _, transform = data_utils.pad_and_scale_img_to_square_img(mask)

            #print(bbox, transform_applied)

            # Transform points
            # Equation: new point = (original point - bbox[low] + pad)*zoom
            coords_to_array = np.linalg.inv(affine) # invert the matrix to convert from points to list indices
            for sl in range(len(streamlines)):
                for i in range(len(streamlines[sl])):
                    x, y, z = streamlines[sl][i]

                    # Convert to array coordinates
                    x, y, z = list(utils.apply_affine(aff=coords_to_array, pts=np.array([[x,y,z]])))[0]

                    # Convert to cropped 144 x 144 x 144 x N coordinates
                    x_new = (x-bbox[0][0]+transform['pad_x'])*transform['zoom']
                    y_new = (y-bbox[1][0]+transform['pad_y'])*transform['zoom']
                    z_new = (z-bbox[2][0]+transform['pad_z'])*transform['zoom']

                    #streamlines[sl][i] = 
                    trk_file.streamlines[sl][i] = np.array([x_new, y_new, z_new])
                    #print("vvv\n%f\t%f\n%f\t%f\n%f\t%f\n^^^" % (x, y, z, x_new, y_new, z_new))
                          
                          
                 
            save_trk(trk_file, 'result.trk', bbox_valid_check=False)

            break
        break

def trk_to_hairnet():
    input_dir  = '../../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms'
    output_dir = '../../../DATASETS/TRACTSEG_105_SUBJECTS/preprocessed/tractograms/1024_streamlines_100_coords_CST'
    subjects = [subject.split('/')[-1] for subject in glob(input_dir + '/*')]

    for subject in subjects:
        if re.match(r'^[0-9]{6}$', subject):
            in_fn = input_dir + '/' + subject + '/tracts/CST_right.trk'
            out_fn = output_dir + '/' + subject + '_CST_right.npy'

            # Load the streamlines
            trk_file = load_trk(in_fn, 'same', bbox_valid_check=False)
            streamlines = trk_file.streamlines

            # Randomly resample to get 1024 streamlines
            streamlines = select_random_set_of_streamlines(streamlines, 1024)

            # Resample to 100 coordinates per streamline
            streamlines = set_number_of_points(streamlines, 100)

            # Convert to numpy array
            streamlines = np.array(streamlines)

            # If fewer than 1024 streamlines, pad with empty streamlines
            if len(streamlines) < 1024:
                temp_streamlines = np.zeros((1024, 100, 3))
                temp_streamlines[:streamlines.shape[0],:streamlines.shape[1], :streamlines.shape[2]] = streamlines
                streamlines = temp_streamlines

            # Reshape
            streamlines = np.reshape(streamlines, (32, 32, 100, 3)) # HairNet shape

            # Save as npy file
            np.save(out_fn, streamlines)

"""
#############
MAIN FUNCTION
#############
"""
#trk_to_hairnet()
