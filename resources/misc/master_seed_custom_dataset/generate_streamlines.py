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
def get_streamlines(seed, mid, ends, points_per_sl, num_sl_to_generate, seed_noise, keypoint_noise, generic_noise, translation):
    result = []
    
    # Generate seed points
    seeds_x = np.random.uniform(low=seed[0]-seed_noise, high=seed[0]+seed_noise, size=num_sl_to_generate)
    seeds_y = np.random.uniform(low=seed[1]-seed_noise, high=seed[1]+seed_noise, size=num_sl_to_generate)
    seeds_z = np.random.uniform(low=seed[2]-seed_noise, high=seed[2]+seed_noise, size=num_sl_to_generate)

    seeds = np.stack([seeds_x, seeds_y, seeds_z], axis=1)
    seeds = np.reshape(seeds, (-1,3))
    
    for i in range(num_sl_to_generate):
        # Generate start, middle, and end points
        mid_offset  = np.random.normal(loc=0, scale=keypoint_noise, size=3)
        end_offset = np.random.normal(loc=0, scale=keypoint_noise, size=3)
        sl = np.array([seeds[i], mid+mid_offset, ends[randint(0,len(ends)-1)]+end_offset])

        # Upscale to desired number of points
        streamlines = set_number_of_points([sl], points_per_sl)
        sl = streamlines[0]

        # Add noise to non-keypoints
        noise = np.random.normal(loc=0, scale=generic_noise, size=sl.shape)
        noise[0], noise[len(noise)//2], noise[-1] = [0,0,0] # don't change the seed points
        sl += noise

        # Add translation to all points
        curr_seed, curr_mid, curr_end = sl[0].copy(), sl[len(sl)//2].copy(), sl[-1].copy()
        t = np.random.uniform(low=-1*translation, high=translation, size=3)
        sl += t

        # Restore the keypoints to before translation
        sl[0], sl[len(sl)//2], sl[-1] = curr_seed, curr_mid, curr_end

        result.append(sl)

    return result

ref_sl_fn = "../../../data/PRE_SAMPLED/not_preprocessed/tractograms/599469_0_CST_left.trk"
_, header = trackvis.read(ref_sl_fn)
#print(header)
#keypoint_noise = 3
#generic_noise = 0.35
#translation= 3

seed_noise = 10
keypoint_noise = 5
generic_noise = 0.23
translation= 1.5
between_subject_noise = 10
#seed_noise = 0.1
#keypoint_noise = 0.03
#generic_noise = 0.005
#translation= 0.001

num_to_gen = int(input("How many tractograms to generate?"))
num_sl_to_generate = int(input("How many streamlines per tractogram?"))
points_per_sl = int(input("How many points per streamline?"))

#start = [0.5, 0.5, 0.05]
#mid = [0.4, 0.4, 0.7]
#ends = [[0.2, 0.2, 0.85], [0.6, 0.6, 0.85]]
start = [-30,-25,-60]
mid = [-25,-30,20]
ends = [[-45,-30,50], [0,15,60]]

for sl_num in range(int(num_to_gen)):
    start_current = start.copy()
    mid_current = mid.copy()
    ends_current = ends.copy()

    # Between subjects noise for seedpoint
    start_current = [s + randint(-1*between_subject_noise, between_subject_noise) for s in start_current]
    mid_current = [s + randint(-1*between_subject_noise, between_subject_noise) for s in mid_current]
    for i in range(len(ends_current)):
        ends_current[i] = [s + randint(-1*between_subject_noise, between_subject_noise) for s in ends_current[i]]

    streamlines = get_streamlines(start_current, mid_current, ends_current, points_per_sl, num_sl_to_generate, seed_noise, keypoint_noise, generic_noise, translation)

    """
    # Visualise the streamlines
    mids = np.array([sl[0] for sl in streamlines])
    mid_x, mid_y, mid_z = mids[:,0], mids[:,1], mids[:,2]
    colors, i = [], 0
    for sl in streamlines:
        colors.extend([i for x in range(len(sl))])
        i += 1
    colors = np.array(colors)
    coords = np.reshape(streamlines, (-1, 3))
    fig = px.line_3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], color=colors, range_x=[-60,20], range_y=[-60,20], range_z=[-70,70])
    #fig.add_trace(px.scatter_3d(x=mid_x, y=mid_y, z=mid_z, color=np.array([1 for i in range(len(mid_x))]), size=[0.5 for item in mid_x]).data[0])
    fig.update_layout(scene_aspectmode='cube')

    fig.show()
    break
    """

    streams = [(sl, None, None) for sl in streamlines]
    trackvis.write('./generated_streamlines/' + str(sl_num) + '.trk', streams, header)
