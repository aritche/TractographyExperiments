#This script uses TractSeg > resources > utilities > trk_2_binary.py
#to generate binary masks for each tract.

echo "*** IMPORTANT NOTICE ***"
echo "trk_2_binary assumes we are using non-legacy data by default. I have edited trk_2_binary to take an extra parameter that defines whether or not we are using legacy data. The file from github may not see this parameter. Double check that the file you are using does use it."
echo "Press any key to continue..."
read dump

# Directory for the trk_2_binary.py script
py_script_dir="../../Custom_Experiment/TractSeg/resources/utility_scripts/trk_2_binary.py"

# File containing the affine information for the HCP data
# Since all HCP data shares the same affine, can use any subject as the source
ref_file="../../DATASETS/HCP_100_SUBJECTS/100307/T1w/Diffusion/nodif_brain_mask.nii.gz"

# Directory for subjects containing tractograms
subjects_dir="../../DATASETS/TRACTSEG_105_SUBJECTS/tractograms"

# Set to 1 if using legacy data (dataset <=V1.1.0), else set to 0
is_legacy=0

# Output directory
output_dir="../../DATASETS/TRACTSEG_105_SUBJECTS/generated_tract_masks"

for subject in `ls -1 $subjects_dir | egrep '^[0-9]{6}$'`
do
    for tract in `ls -1 ${subjects_dir}/${subject}/tracts`
    do
        echo $tract
        tractogram="${subjects_dir}/${subject}/tracts/$tract"

        subject_output="${output_dir}/${subject}"
        mkdir $subject_output
        python3 $py_script_dir $tractogram ${subject_output}/${tract}.nii.gz $ref_file $is_legacy
    done
done
