"""
This file is a modified/simplified version of TractSeg > tractseg > data > preprocessing.py
https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/data/preprocessing.py

It will take in 145 x 174 x 144 tract masks, TOMs, and endings masks, crop them and pad them to
achieve a square size of 144 x 144 x 144
"""

import nibabel as nib
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy import ndimage
import psutil
from joblib import Parallel, delayed

"""
Potential concerns:
- Is it better to crop according to nodif_brain_mask, or should each
  tract be cropped independently with its own bounding box?
- What is the function to return square images back to normal?
"""

def resize_first_three_dims(img, order=0, zoom=0.62, nr_cpus=-1):
    # THIS FUNCTION IS FROM TractSeg > tractseg > libs > img_utils.py
    # https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/libs/img_utils.py
    def _process_gradient(grad_idx):
        return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

    nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
    img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
    return np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back

def pad_and_scale_img_to_square_img(data, target_size=144, nr_cpus=-1):
    # THIS FUNCTION IS FROM TractSeg > tractseg > libs > data_utils.py
    # https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/libs/data_utils.py
    """
    Expects 3D or 4D image as input.

    Does
    1. Pad image with 0 to make it square
        (if uneven padding -> adds one more px "behind" img; but resulting img shape will be correct)
    2. Scale image to target size
    """
    nr_dims = len(data.shape)
    assert (nr_dims >= 3 and nr_dims <= 4), "image has to be 3D or 4D"

    shape = data.shape
    biggest_dim = max(shape)

    # Pad to make square
    if nr_dims == 4:
        new_img = np.zeros((biggest_dim, biggest_dim, biggest_dim, shape[3])).astype(data.dtype)
    else:
        new_img = np.zeros((biggest_dim, biggest_dim, biggest_dim)).astype(data.dtype)
    pad1 = (biggest_dim - shape[0]) / 2.
    pad2 = (biggest_dim - shape[1]) / 2.
    pad3 = (biggest_dim - shape[2]) / 2.
    new_img[int(pad1):int(pad1) + shape[0],
            int(pad2):int(pad2) + shape[1],
            int(pad3):int(pad3) + shape[2]] = data

    # Scale to right size
    zoom = float(target_size) / biggest_dim
    if nr_dims == 4:
        #use order=0, otherwise does not work for peak images (results would be wrong)
        new_img = resize_first_three_dims(new_img, order=0, zoom=zoom, nr_cpus=nr_cpus)
    else:
        new_img = ndimage.zoom(new_img, zoom, order=0)

    transformation = {
        "original_shape": shape,
        "pad_x": pad1,
        "pad_y": pad2,
        "pad_z": pad3,
        "zoom": zoom
    }

    return new_img, transformation

def get_bbox_from_mask(mask, outside_value=0):
    # THIS FUNCTION IS FROM TractSeg > tractseg > libs > data_utils.py
    # https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/libs/data_utils.py
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_nonzero(data, bbox=None):
    # THIS FUNCTION IS FROM TractSeg > tractseg > libs > data_utils.py
    # https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/libs/data_utils.py
    # I have adjusted it to squeeze the output
    cropped_data = []
    for c in range(data.shape[3]):
        cropped = crop_to_bbox(data[:,:,:,c], bbox)
        cropped_data.append(cropped)
    data = np.squeeze(np.array(cropped_data).transpose(1,2,3,0))

    return data

def crop_to_bbox(image, bbox):
    # THIS FUNCTION IS FROM TractSeg > tractseg > libs > data_utils.py
    # https://github.com/MIC-DKFZ/TractSeg/blob/master/tractseg/libs/data_utils.py
    assert len(image.shape) == 3, "only supports 3d images"
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

mask = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/hcp_brain_masks/672756/nodif_brain_mask.nii.gz').get_data()
bbox = get_bbox_from_mask(np.nan_to_num(mask), 0)
print(bbox)
img = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/generated_toms/672756/CST_left_DIRECTIONS.nii.gz')
#img = nib.load('../../DATASETS/tractseg_output_672756/TOM/CST_left.nii.gz')
affine = img.affine
data = img.get_data()
print(data.shape)
if len(data.shape) == 3:
    data = data[..., None]
data = crop_to_nonzero(np.nan_to_num(data), bbox=bbox)
data, t = pad_and_scale_img_to_square_img(data)
print(data.shape)

nib.save(nib.Nifti1Image(data, affine), "github_ref.nii.gz")

"""
#fn = '../../DATASETS/HCP_100_SUBJECTS/100307/T1w/Diffusion/nodif_brain_mask.nii.gz' # nodif_brain_mask
base_dir = '../../DATASETS/TRACTSEG_105_SUBJECTS'
subjects = [subject.split('/')[-1] for subject in glob(base_dir + '/hcp_brain_masks/*')]

for subject in subjects:
    # Compute the bounding box from the whole brain mask
    mask_fn = base_dir + '/hcp_brain_masks/' + subject + '/nodif_brain_mask.nii.gz'
    data = nib.load(mask_fn).get_data()
    bbox = get_bbox_from_mask(np.nan_to_num(data), 0)

    for file_type in ['generated_tract_masks', 'generated_toms', 'generated_endings_masks']:
        fn = base_dir + '/' + file_type + '/' + subject + '/*'
        for tract_fn in glob(fn):
            # Load the file
            img = nib.load(tract_fn)
            data = img.get_data()
            affine = img.affine
            data = np.nan_to_num(data)

            # Add channel dimension if does not exist yet
            if len(data.shape) == 3:
                data = data[..., None]

            # Crop using the existing bounding box
            data = crop_to_nonzero(data, bbox=bbox)
            data, _ = pad_and_scale_img_to_square_img(data)
            print(data.shape)

            # Write to file
            nib.save(nib.Nifti1Image(data, affine), "test_" + tract_fn.split('/')[-1])
            break
        break
    break
"""
