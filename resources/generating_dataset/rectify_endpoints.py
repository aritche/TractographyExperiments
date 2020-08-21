# A script to match streamlines to a set of beginning and endpoints.
# i.e. reverse a streamline if it starts at an endpoint and ends at a beginning.
import nibabel as nib
from nibabel import trackvis
import numpy as np
import sys
from sklearn.cluster import KMeans

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt

def rectify(tractogram_fn, out_fn):
    # Open file and extract streamlines
    streams, header = trackvis.read(tractogram_fn)
    streamlines = [s[0] for s in streams]

    # Get extrema
    beginnings = [s[0] for s in streamlines]
    #endings = [s[-1] for s in streamlines]

    # Run KMeans to cluster beginnings and endings both into 2 clusters
    kmeans_beginnings = KMeans(n_clusters = 2).fit(beginnings)
    #kmeans_endings = KMeans(n_clusters = 2).fit(endings)

    beginnings_centers = kmeans_beginnings.cluster_centers_
    #endings_centers = kmeans_endings.cluster_centers_

    # Find the axis with the largest difference between extreme coordinates and 
    diff_X = abs(beginnings_centers[0][0] - beginnings_centers[1][0])
    diff_Y = abs(beginnings_centers[0][1] - beginnings_centers[1][1])
    diff_Z = abs(beginnings_centers[0][2] - beginnings_centers[1][2])

    # Assign the cluster with the largest value for this axis to be the seed cluster
    max_val = max(diff_X, diff_Y, diff_Z)
    if diff_X == max_val:
        max_index = 0
    elif diff_Y == max_val:
        max_index = 1
    else:
        max_index = 2
    if beginnings_centers[0][max_index] > beginnings_centers[1][max_index]:
        beginnings_beginnings = 0
        beginnings_endings = 1
    else:
        beginnings_beginnings = 1
        beginnings_endings = 0

    """
    # Assign the more populous cluster for beginnings to be the beginnings
    pred_beginnings = kmeans_beginnings.predict(beginnings)
    beginnings_sum = np.sum(pred_beginnings) / len(pred_beginnings)
    if beginnings_sum > 0.5:
        #beginnings_beginnings = beginnings_centers[1]
        #beginnings_endings = beginnings_centers[0]
        beginnings_beginnings = 1
        beginnings_endings = 0
    else:
        #beginnings_beginnings = beginnings_centers[0]
        #beginnings_endings = beginnings_centers[1]
        beginnings_beginnings = 0
        beginnings_endings = 1
    """

    # Assign the more populous cluster for endings to be the endings
    #pred_endings = kmeans_endings.predict(endings)
    #endings_sum = np.sum(pred_endings) / len(pred_endings)
    #if endings_sum > 0.5:
    #    endings_endings = 1
    #    endings_beginnings = 0
    #else:
    #    endings_endings = 0
    #    endings_beginnings = 1

    # For each streamline, run kmeans.fit() on the beginning point and reverse it if appropriate
    new_streamlines = []
    for sl in streamlines:
        seed = np.array([sl[0]], dtype=float)
        result = kmeans_beginnings.predict(seed)

        # If assigned to beginnings_endings, reverse the streamline
        if result[0] == beginnings_endings:
            reversed_sl = sl[::-1]
            new_streamlines.append(reversed_sl)
        else:
            new_streamlines.append(sl)

    #fig = plt.figure()
    #ax = Axes3D(fig)
    #seeds = np.array([s[0] for s in new_streamlines])
    #x, y, z = seeds[:,0], seeds[:,1], seeds[:,2]
    #ax.scatter(list(x), list(y), list(z), c='#003cff')

    #seeds = np.array([s[-1] for s in new_streamlines])
    #x, y, z = seeds[:,0], seeds[:,1], seeds[:,2]
    #ax.scatter(list(x), list(y), list(z), c='#aabbff')
    #plt.show()


    new_streamlines = [(s, None, None) for s in new_streamlines]
    trackvis.write(out_fn, streamlines=new_streamlines, hdr_mapping=header)

    #return new_streamlines
    
in_tractogram_fn = sys.argv[1]
out_tractogram_fn = sys.argv[2]
rectify(in_tractogram_fn, out_tractogram_fn)
