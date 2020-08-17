# Generate a seed volume from a tractogram. Results will be in unprocessed space, i.e. 
# the result of this program will still need to be processed (i.e. cropped/zoomed)
import sys
import nibabel as nib
from dipy.io.image import load_nifti 
from dipy.tracking import utils
import numpy as np

def seeds_to_vol(trk_fn, tom_fn, out_fn, reverse):
    # Open references volume to get the affine transformation
    ref_data, ref_affine = load_nifti(tom_fn)

    # Invert the reference affine transform to go from coordinates to voxels
    inverted_ref_affine = np.linalg.inv(ref_affine)

    # Open tract you want to convert to volume
    tractogram = nib.streamlines.load(trk_fn)
    streamlines = tractogram.streamlines
    streamlines = np.array(streamlines)

    # Get the seeds
    #seeds = streamlines[:,0,:]
    if reverse == 0:
        seeds = [s[0] for s in streamlines]
    else:
        seeds = [s[-1] for s in streamlines]

    # Create a template output volume
    result = np.zeros((145,174,145))

    # Set seed points to 1
    for coord in seeds:
        x, y, z = list(utils.apply_affine(aff=inverted_ref_affine, pts=np.array([coord])))[0]
        x, y, z = int(x), int(y), int(z)
        result[x][y][z] = 1

    # Save result
    nifti_result = nib.Nifti1Image(result.astype("uint8"), ref_affine)
    nib.save(nifti_result, out_fn)

trk_fn = sys.argv[1]
tom_fn = sys.argv[2]
out_fn = sys.argv[3]
reverse = int(sys.argv[4])
seeds_to_vol(trk_fn, tom_fn, out_fn, reverse)
