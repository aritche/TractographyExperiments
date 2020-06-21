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
import nibabel as nib

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
    ref_data, ref_affine = load_nifti(ref_file)
    coords_to_array = np.linalg.inv(ref_affine) # invert the matrix to convert from points to list indices

    #print(utils.apply_affine(aff=np.linalg.inv(ref_affine), pts=np.array([[0,0,0]])))

   
    # coordinates
    a = []
    b = []
    c = []

    # vectors
    d = []
    e = []
    f = []

    i = 0

    collection = {}
    collection = []
    collection = [[[[np.array([0,0,0])] for x in range(len(ref_data[z][y]))] for y in range(len(ref_data[z]))] for z in range(len(ref_data))]
    for sl in streamlines:
        for point in range(len(sl)-1):
            if i % 10 == 0:
                x, y, z = sl[point]

                # Convert from (0,0,0) = centre, to (0,0,0) = top left
                x, y, z = list(utils.apply_affine(aff=np.linalg.inv(ref_affine), pts=np.array([[x,y,z]])))[0]

                # Compute direction of movement
                vector = np.array(sl[point+1]) - np.array(sl[point])

                # Normalise the magnitude
                size = (vector[0]**2 + vector[1]**2 + vector[2]**2)**(1/2)
                u, v, w = vector / size if size != 0 else [0, 0, 0]

                x, y, z = [int(x), int(y), int(z)]
                collection[z][y][x].append(np.array([u, v, w]))
                #if (z, y, x) in collection:
                #    collection[(z, y, x)].append((u, v, w))
                #else:
                #    collection[(z, y, x)] = [(u, v, w)]
                #collection[z, y, x]
                #a.append(x)
                #b.append(y)
                #c.append(z)
                #d.append(u)
                #e.append(v)
                #f.append(w)
            i += 1

    # get means
    for z in range(len(collection)):
        for y in range(len(collection[z])):
            for x in range(len(collection[z][y])):
                collection[z][y][x] = sum(collection[z][y][x])/len(collection[z][y][x])

    collection = np.array(collection)

    for z in range(len(collection)):
        im = collection[z]
        #cv2.imshow('TOM', np.uint8(255*(im - np.min(im))/(np.max(im) - np.min(im))))
        cv2.imshow('TOM', np.uint8(im*255))
        cv2.waitKey(0)
    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter(a, b, c)
    #plt.show()

    #print(min(a), max(a), abs(max(a)-min(a)))
    #print(min(b), max(b), abs(max(b)-min(b)))
    #print(min(c), max(c), abs(max(c)-min(c)))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.quiver(a, b, c, d, e, f)
    #plt.show()


#sl = load_tracts('../../TractSeg_Replication/concatenated_result.trk')
#sl2 = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/MCP.trk')
#sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/CST_right.trk')
sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/CST_left.trk')
#sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/CA.trk')
#sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/FX_left.trk')
#sl = load_tracts('../../DATASETS/TRACTSEG_105_SUBJECTS/672756/tracts/FX_right.trk')
ref_file = '../../DATASETS/HCP_100_SUBJECTS/672756/T1w/Diffusion/nodif_brain_mask.nii.gz'
gen_TOM(sl, ref_file)
