import sys
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
import numpy as np
import plotly.express as px
from random import randint
from nibabel import trackvis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# Given three (x,y,z) coordinates, generate num_sl streamlines that pass through these points
def get_streamlines(seed, mid, ends, num_sl, points_per_sl):
    result = []
    for i in range(num_sl):
        # Generate start, middle, and end points
        seed_offset = np.random.normal(loc=0, scale=3, size=3)
        mid_offset  = np.random.normal(loc=0, scale=3, size=3)
        end_offset = np.random.normal(loc=0, scale=3, size=3)
        sl = np.array([seed+seed_offset, mid+mid_offset, ends[randint(0,len(ends)-1)]+end_offset])

        # Upscale to desired number of points
        streamlines = set_number_of_points([sl], points_per_sl)
        sl = streamlines[0]

        # Add noise to non-keypoints
        noise = np.random.normal(loc=0, scale=0.35, size=sl.shape)
        noise[0], noise[len(noise)//2], noise[-1] = [0,0,0] # don't change the seed points
        sl += noise

        # Add translation to all points
        curr_seed, curr_mid, curr_end = sl[0], sl[len(sl)//2], sl[-1]
        t = np.random.uniform(low=-5, high=5, size=3)
        sl += t

        # Restore the keypoints to before translation
        sl[0], sl[len(sl)//2], sl[-1] = curr_seed, curr_mid, curr_end

        result.append(sl)

    fn = "../../../data/PRE_SAMPLED/tractograms/599469_0_CST_left.trk"
    _, header = trackvis.read(fn)
    streams = [(sl, None, None) for sl in result]
    trackvis.write('result1.trk', streams, header)

    streamlines = result
    coords = np.reshape(streamlines, (-1, 3))
    x, y, z = coords[:,0], coords[:,1], coords[:,2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(list(x), list(y), list(z))
    ax.set_xlim([-50, 0])
    ax.set_ylim([-40, 0])
    ax.set_zlim([-60, 70])
    plt.show()

    #print(x)

    colors, i = [], 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    #fig = px.line_3d(x=x, y=y, z=z, color=colors, range_x=[-50,10], range_y=[-50,10], range_z=[-60,60])
    fig = px.line_3d(x=x, y=y, z=z, color=colors)
    #fig.add_trace(px.scatter_3d(x=x, y=y, z=z, color=np.array([1 for i in range(len(x))])).data[0])
    fig.show()

    """
    ref_sl = np.array([seed, mid, end])
    streamlines = [ref_sl]
    streamlines = set_number_of_points(streamlines, points_per_sl)
    ref_sl = streamlines[0]

    for i in range(num_sl):
        noise = np.random.normal(loc=0, scale=0.35, size=ref_sl.shape)
        new_sl = ref_sl + noise
        t = np.random.uniform(low=-5, high=5, size=3)
        t[-1] = 0
        new_sl += t

        new_sl[0] = ref_sl[0]
        new_sl[-1] = ref_sl[-1]
        new_sl[len(new_sl)//2] = ref_sl[len(ref_sl)//2]

        streamlines.append(new_sl)

    coords = np.reshape(streamlines, (-1, 3))
    x, y, z = coords[:,0], coords[:,1], coords[:,2]
    #print(x)

    colors, i = [], 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)

    #fig = px.line_3d(x=x, y=y, z=z, color=colors, range_x=[-50,10], range_y=[-50,10], range_z=[-60,60])
    fig = px.line_3d(x=x, y=y, z=z, color=colors)
    #fig.add_trace(px.scatter_3d(x=x, y=y, z=z, color=np.array([1 for i in range(len(x))])).data[0])
    fig.show()
    """
get_streamlines([-30,-35,-50], [-20,-25,15], [[-20,-10,55], [-5,-30,55]], 300, 40)

