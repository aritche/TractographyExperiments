#This script uses TractSeg > resources > utilities > create_endpoints_mask_with_clustering.py
#to generate binary masks for the start and end of each tract.

echo "*** IMPORTANT NOTICE ***"
echo "create_endpoints_mask_with_clustering.py assumes we are using legacy data by default. I have edited trk_2_binary to take an extra parameter that defines whether or not we are using legacy data. Double check that the file you are using does use it."
echo "Press any key to continue..."
read dump

# Directory for the create_endpoints_mask_with_clustering.py script
py_script_dir="../../Custom_Experiment/TractSeg/tractseg/libs/create_endpoints_mask_with_clustering.py"

# File containing the affine information for the HCP data
# Since all HCP data shares the same affine, can use any subject as the source
ref_file="../../DATASETS/HCP_100_SUBJECTS/100307/T1w/Diffusion/nodif_brain_mask.nii.gz"

# Directory for subjects containing tractograms
subjects_dir="../../DATASETS/V1.1.0_TRACTSEG_105_SUBJECTS_V1.1.0"

# Output directory
output_dir="../../DATASETS/TRACTSEG_105_SUBJECTS/generated_endings_masks"

for subject in `ls -1 $subjects_dir | egrep '^[0-9]{6}$'`
do
    echo $subject

    subject_output="${output_dir}/${subject}"
    mkdir $subject_output

    for tract in `ls -1 ${subjects_dir}/${subject}/tracts | sed 's/\.trk//g' `
    do
        tractogram="${subjects_dir}/${subject}/tracts/${tract}.trk"

        python3 $py_script_dir $ref_file $tractogram ${subject_output}/${tract}
    done
done
