from nibabel import trackvis
import numpy as np
import cv2
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines

points_per_sl = 15

fn = '../../data/256_25_CST_left/not_preprocessed/tractograms/599469_0_CST_left.trk'
streamline_data, header = trackvis.read(fn)
streamlines = [s[0] for s in streamline_data]
forward_points = set_number_of_points(streamlines, points_per_sl)
backward_points = [item[::-1] for item in forward_points]


fiber_maps = []
for i in range(len(streamlines)):
    rowA = np.append(forward_points[i], backward_points[i], axis=0)
    rowB = np.append(backward_points[i], forward_points[i], axis=0)
    fiber_map = np.zeros((points_per_sl*2,points_per_sl*2,3))
    for row in range(points_per_sl*2):
        if (row % 2 == 0):
            fiber_map[row] = rowA
        else:
            fiber_map[row] = rowB
    fiber_maps.append(fiber_map)

cv2.namedWindow('map', cv2.WINDOW_NORMAL)
for fiber_map in fiber_maps:
    #abs_map = np.abs(fiber_map)
    abs_map = fiber_map
    abs_map = (abs_map - np.min(abs_map)) / (np.max(abs_map) - np.min(abs_map))
    abs_map *= 255
    cv2.imshow('map', np.uint8(abs_map))
    cv2.waitKey(0)
