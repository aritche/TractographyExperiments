"""
File for reading / experimenting with nifti files
"""

import nibabel as nib

ref_file = '../../DATASETS/HCP_100_SUBJECTS/672756/T1w/Diffusion/nodif_brain_mask.nii.gz'

data = nib.load(ref_file)
print(data.header)
