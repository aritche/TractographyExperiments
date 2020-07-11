"""
NOTE: THIS FILE ONLY WORKS WITH <=V1.1.0 OF THE ZENODO TRACTSEG DATASET

Given a tractogram, this file will generate 'offspring' tractograms that
are random sub-samples of the streamlines of the input tractogram, e.g.
if the input has 5983 streamlines, the program can generate N new tractograms
of X streamlines by randomly sampling N times from the input
"""
import dipy
from nibabel import trackvis
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from dipy.io.streamline import load_trk
import numpy as np
import random
import sys

def get_offspring(tractogram_fn, num_sl_to_sample, output_dir, subject, tract_name):
    # Load (using old TractSeg dataset <=V1.1.0 https://github.com/MIC-DKFZ/TractSeg/blob/17b33b37bafad7566de6372a534a14a1ef5a7384/resources/utility_scripts/trk_2_binary.py)
    streams, header = trackvis.read(tractogram_fn)

    # Calculate how many samples to take
    num_offspring = len(streams) // num_sl_to_sample
    num_offspring = num_offspring + 1 if num_offspring == 0 else num_offspring

    for i in range(num_offspring):
        new_streams = random.sample(streams, num_sl_to_sample)

        # Note, new header not needed since # of streamlines is automatically updated when the file is saved
        trackvis.write(output_dir + "/" + subject + "_"+ str(i) + "_" + tract_name +".trk", streamlines=new_streams, hdr_mapping=header)

input_fn = sys.argv[1]
output_dir = sys.argv[2]
subject = sys.argv[3] # prefix for the output file (e.g. a subject number)
tract_name = sys.argv[4] # prefix for the output file (e.g. a subject number)
get_offspring(input_fn, 1024, output_dir, subject, tract_name)
