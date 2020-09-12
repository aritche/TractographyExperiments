"""
NOTE: THIS FILE ONLY WORKS WITH <=V1.1.0 OF THE ZENODO TRACTSEG DATASET

Given a trk file, this will make it such that each streamline has a fixed
number of points in it (equally spaced) 
"""
from nibabel import trackvis
from dipy.tracking.streamline import set_number_of_points
import sys

def get_offspring(tractogram_fn, output_dir, subject, tract_name, points_per_sl):
    # Load (using old TractSeg dataset <=V1.1.0 https://github.com/MIC-DKFZ/TractSeg/blob/17b33b37bafad7566de6372a534a14a1ef5a7384/resources/utility_scripts/trk_2_binary.py)
    streams, header = trackvis.read(tractogram_fn)

    # Re-sample each streamline to have the required number of points
    # Since trackvis.read returns a tuple for each streamline (with streamline coordinates being
    #  the first element in that list, first we need to extract those actual coordinates)
    only_streamlines = [sl[0] for sl in streams] # extract the actual coordinates
    resampled_points = set_number_of_points(only_streamlines, points_per_sl)
    
    # Now store these resampled coordinates into the streamline tuples
    # Since you can't modify tuples, we need to re-write the tuples entirely
    for sl_num in range(len(streams)):
        streams[sl_num] = (resampled_points[sl_num], streams[sl_num][1], streams[sl_num][2])

    # Note, new header not needed since # of streamlines is automatically updated when the file is saved
    trackvis.write(output_dir + "/" + subject + "_" + tract_name + ".trk", streamlines=streams, hdr_mapping=header)

if len(sys.argv) == 6:
    input_fn = sys.argv[1]
    output_dir = sys.argv[2]
    subject = sys.argv[3] # prefix for the output file (e.g. a subject number)
    tract_name = sys.argv[4] # prefix for the output file (e.g. a subject number)
    points_per_sl = int(sys.argv[5])
    get_offspring(input_fn, output_dir, subject, tract_name, points_per_sl)
else:
    print("ERROR: Incorrect number of arguments.")
