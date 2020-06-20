import dipy
import torch
from dipy.io.image import load_nifti, load_nifti_data
from dipy.io.streamline import load_trk
from dipy.segment.metric import ResampleFeature
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from dipy.tracking import utils
from dipy.tracking.life import voxel2streamline
import cv2

# Function to load a tractogram into a numpy array
def load_tracts(fn):
    print('Loading streamlines...')

    # Loaded into 'rasmm' space by default
    # According to: https://nipy.org/nibabel/reference/nibabel.trackvis.html
    #               'rasmm' = "points are expressed in mm space according to the affine."
    # TODO PROBLEM: ************* NOT SURE IF I SHOULD BE USING 'RASMM' or 'VOXMM'
    tract = load_trk(fn, 'same', bbox_valid_check=False)
    streamlines = tract.streamlines
    return streamlines

# Given a tractogram numpy array, resample each streamline to contain 'n' equally spaced points
# Tutorial: https://dipy.org/documentation/1.1.1./examples_built/segment_clustering_features/
def resample_tracts(t, n):
    feature = ResampleFeature(nb_points=n)
    print(dir(feature))


def gen_TOM(streamlines, ref_file):
    # Using an approach adapted from https://github.com/MIC-DKFZ/TractSeg/blob/master/resources/utility_scripts/trk_2_binary.py 
    data, ref_affine, axes = load_nifti(ref_file, return_coords=True)
    ref_shape = data.shape

    density_map = utils.density_map(streamlines=streamlines, vol_dims=ref_shape, affine=ref_affine)
    #print(density_map)
    #print(density_map.shape)

    #result = voxel2streamline(streamlines, affine=ref_affine)
    for sl in streamlines:
        #print("Affine:")
        #print(ref_affine)
        #print("------")
        #print(sl[0])
        result = utils.apply_affine(aff=ref_affine, pts=sl)
        #for i in range(len(sl)):
        #    print(sl[i], result[i])
        break
    return 
    


    #print(streamlines[0,:].shape)
    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    i = 0

    collection = {}
    for sl in streamlines:
        for point in range(len(sl)-1):
            if i % 1000 == 0:
                x, y, z = sl[point]
                vector = np.array(sl[point+1]) - np.array(sl[point])
                size = (vector[0]**2 + vector[1]**2 + vector[2]**2)**(1/2)
                if size != 0:
                    u, v, w = vector / size
                else:
                    u, v, w = 0
                a.append(x)
                b.append(y)
                c.append(z)
                d.append(u)
                e.append(v)
                f.append(w)
            i += 1

    """
    for sl in sl2:
        for point in range(len(sl)-1):
            if i % 1000 == 0:
                x, y, z = sl[point ]
                nx, ny, nz = sl[point+1]
                vs = (nx **2 + ny **2 + nz**2)**(1/2)
                nx, ny, nz = [nx/vs, ny/vs, nz/vs]
                a.append(x)
                b.append(y)
                c.append(z)
                d.append(nx-x)
                e.append(ny-y)
                f.append(nz-z)
            i += 1
        #print(sl.shape)
    """

    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(a, b, c)
    #plt.show()

    print(min(a), max(a), abs(max(a)-min(a)))
    print(min(b), max(b), abs(max(b)-min(b)))
    print(min(c), max(c), abs(max(c)-min(c)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(a, b, c, d, e, f)
    plt.show()


#sl = load_tracts('../../TractSeg_Replication/concatenated_result.trk')
#sl2 = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/MCP.trk')
sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/CST_right.trk')
ref_file = '../../DATASETS/HCP_100_SUBJECTS/672756/T1w/Diffusion/nodif_brain_mask.nii.gz'
gen_TOM(sl, ref_file)
