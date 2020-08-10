import nibabel as nib
from dipy.io.streamline import load_trk
from dipy.io.image import load_nifti, load_nifti_data
from dipy.tracking import utils
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Open references volume to get the affine transformation
volume_fn = '/media/aritche/1TB HDD Volume/PHD/TractographyExperiments/data/1024_40_CST_left_fixed/preprocessed/TOMs/601127_0_CST_left.nii.gz'
ref_data, ref_affine = load_nifti(volume_fn)

# Invert the reference affine transform to go from coordinates to voxels
inverted_ref_affine = np.linalg.inv(ref_affine)

# Open tract you want to convert to volume
tracts_fn = '/media/aritche/1TB HDD Volume/PHD/TractographyExperiments/data/1024_40_CST_left_fixed/not_preprocessed/tractograms/601127_0_CST_left.trk' 
tractogram = nib.streamlines.load(tracts_fn)
streamlines = tractogram.streamlines
streamlines = np.array(streamlines)

# Get the seeds
seeds = streamlines[:,0,:]

# Create a template output volume
result = np.zeros((144,144,144))

# Set seed points to 1
for coord in seeds:
    x, y, z = list(utils.apply_affine(aff=inverted_ref_affine, pts=np.array([coord])))[0]
    x, y, z = int(x), int(y), int(z)
    result[x][y][z] = 1

# Save result
nifti_result = nib.Nifti1Image(result.astype("uint8"), ref_affine)
nib.save(nifti_result, 'result.nii.gz')
