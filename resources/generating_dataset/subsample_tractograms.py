"""
NOTE: THIS FILE ONLY WORKS WITH <=V1.1.0 OF THE ZENODO TRACTSEG DATASET

Given a tractogram, this file will generate 'offspring' tractograms that
are random sub-samples of the streamlines of the input tractogram, e.g.
if the input has 5983 streamlines, the program can generate N new tractograms
of X streamlines by randomly sampling N times from the input. It will also
re-sample each streamline to have a fixed number of points.
"""
import dipy
from nibabel import trackvis
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from dipy.io.streamline import load_trk
import numpy as np
import random
import sys

def get_offspring(tractogram_fn, num_sl_to_sample, output_dir, subject, tract_name, points_per_sl):
    # Load (using old TractSeg dataset <=V1.1.0 https://github.com/MIC-DKFZ/TractSeg/blob/17b33b37bafad7566de6372a534a14a1ef5a7384/resources/utility_scripts/trk_2_binary.py)
    streams, header = trackvis.read(tractogram_fn)

    # Calculate how many samples to take
    #print("%d streamlines in original file" % (len(streams)))
    num_offspring = int(len(streams) / num_sl_to_sample * 1.5)
    num_offspring = num_offspring + 1 if num_offspring == 0 else num_offspring

    for i in range(num_offspring):
        # Sample the required number of streamlines
        if len(streams) >= num_sl_to_sample:
            new_streams = random.sample(streams, num_sl_to_sample)
        else: # If not enough streamlines to sample from, then just pad with all (0,0,0) streamlines
            new_streams = streams
            while len(new_streams) < num_sl_to_sample:
                new_streams.append((np.zeros((40,3),dtype=np.float32), None, None))

        # Re-sample each streamline to have the required number of points
        # Since trackvis.read returns a tuple for each streamline (with streamline coordinates being
        #  the first element in that list, first we need to extract those actual coordinates)
        only_streamlines = [sl[0] for sl in new_streams] # extract the actual coordinates
        resampled_points = set_number_of_points(only_streamlines, points_per_sl)
        
        # Now store these resampled coordinates into the streamline tuples
        # Since you can't modify tuples, we need to re-write the tuples entirely
        for sl_num in range(len(new_streams)):
            new_streams[sl_num] = (resampled_points[sl_num], new_streams[sl_num][1], new_streams[sl_num][2])

        # If fewer than the required number of streamlines can be sampled, pad with "0" streamlines
        diff = num_sl_to_sample - len(new_streams)
        if diff > 0:
            print("Not enough streamlines to sample, so generating empty streamlines for padding.")
            empty_sl = np.zeros((points_per_sl, 3), dtype=np.float32)
            for num in range(diff):
                new_streams.append(empty_sl)
            
        # Note, new header not needed since # of streamlines is automatically updated when the file is saved
        trackvis.write(output_dir + "/" + subject + "_"+ str(i) + "_" + tract_name +".trk", streamlines=new_streams, hdr_mapping=header)

if len(sys.argv) == 7:
    input_fn = sys.argv[1]
    output_dir = sys.argv[2]
    subject = sys.argv[3] # prefix for the output file (e.g. a subject number)
    tract_name = sys.argv[4] # prefix for the output file (e.g. a subject number)
    num_streamlines_per_tract = int(sys.argv[5])
    points_per_sl = int(sys.argv[6])
    get_offspring(input_fn, num_streamlines_per_tract, output_dir, subject, tract_name, points_per_sl)
else:
    print("ERROR: Incorrect number of arguments.")
