from nibabel import trackvis
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric

import numpy as np

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#from plotly import graph_objs as go
#import plotly.express as px
import sys

def plot_streamlines_plotly(streamlines):
    colors = []
    i = 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    coords = np.reshape(np.array(streamlines), (-1, 3))
    fig = px.line_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], color=colors)
    fig.show()

args = sys.argv
tractogram_fn = args[1]
output_dir = args[2]
subject = args[3]
tract_name = args[4]
max_num_centroids = int(args[5])
points_per_sl = int(args[6])

# Open the file and extract streamlines
streams, header = trackvis.read(tractogram_fn)
streamlines = [sl[0] for sl in streams]

# Run quickbundles with chosen parameters
feature = ResampleFeature(nb_points=points_per_sl)
metric = AveragePointwiseEuclideanMetric(feature)
qb = QuickBundles(threshold=10., max_nb_clusters=max_num_centroids, metric=metric)
clusters = qb.cluster(streamlines)

# Extract the centroids
centroids = [cluster.centroid for cluster in clusters]

# If not enough generated, fill with empty streamlines
diff = max_num_centroids - len(centroids)
if diff > 0:
    print("Not enough centroids generated, so generating empty streamlines for padding.")
    empty_sl = np.zeros((points_per_sl, 3), dtype=np.float32)
    for num in range(diff):
        centroids.append(empty_sl)

# Convert to TrackVis format and write to file
centroids = [(c, None, None) for c in centroids]
out_fn = output_dir + '/' + subject + '_' + tract_name + '.trk'
trackvis.write(out_fn, centroids, header)
