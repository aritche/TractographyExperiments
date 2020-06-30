from tractseg.libs import data_utils # assumes tractseg is installed
# ALTERNATIVELY, you could import it directly if you download the file from TractSeg > tractseg > libs > data_utils.py
# In this case, you would also need TractSeg > tractseg > libs > img_utils.py located in the same directory
# import data_utils

import nibabel as nib
import numpy as np

# Load the nodif_brain_mask and get the bounding box of the brain region
mask = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/hcp_brain_masks/672756/nodif_brain_mask.nii.gz').get_data()
bbox = data_utils.get_bbox_from_mask(np.nan_to_num(mask), 0)
print(bbox)

# Load the file you are interested in preprocessing
#img = nib.load('../../DATASETS/tractseg_output_672756/TOM/CST_left.nii.gz')
#img = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/generated_toms/672756/CST_left_DIRECTIONS.nii.gz')
img = nib.load('../../DATASETS/TRACTSEG_105_SUBJECTS/generated_tract_masks/672756/CST_left.trk.nii.gz')
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
