echo "THIS PROGRAM ONLY WORKS FOR TRACTSEG TRACTOGRAMS <=V1.1.0. You need to change the way some of the helper files open .trk files to use this script on more recent tractograms."
echo "Press any key to confirm..."
read dump

# Locate all of the relevant scripts
quickbundles_script="./interpolation/quickbundles.py"
generate_masks="./tractseg_helper_scripts/trk_2_binary.py"
generate_endings="./tractseg_helper_scripts/create_endpoints_mask_with_clustering.py"
generate_toms="../../../../Programs/MITK-Diffusion-Ubuntu/MitkFiberDirectionExtraction.sh"
preprocess="../IO/preprocess.py"

input_base="../../../DATASETS/V1.1.0_TRACTSEG_105_SUBJECTS_V1.1.0"

output_base="./generated_datasets"
output_trk_base="${output_base}/tractograms"
output_masks_base="${output_base}/tract_masks"
output_endings_base="${output_base}/endings_masks"
output_toms_base="${output_base}/TOMs"

preprocessed_masks="${output_base}/preprocessed/tract_masks"
preprocessed_endings="${output_base}/preprocessed/endings_masks"
preprocessed_beginnings="${output_base}/preprocessed/beginnings_masks"
preprocessed_toms="${output_base}/preprocessed/TOMs"

mkdir $output_base
mkdir $output_trk_base
mkdir $output_masks_base
mkdir $output_endings_base
mkdir $output_toms_base
mkdir ${output_base}/preprocessed
mkdir $preprocessed_masks
mkdir $preprocessed_endings
mkdir $preprocessed_beginnings
mkdir $preprocessed_toms

# Sub-sample tractograms, and output to $output_dir
echo "What is the maximum number of quickbundle centroids you want to generate?"
read num_sl
echo "How many points do you want per streamline?"
read points_per_sl
echo "Which tract do you want to process?"
read tract_name
for subject in `ls -1 $input_base | egrep '^[0-9]{6}$'`
do
    echo $subject
    trk_fn="${input_base}/${subject}/tracts/${tract_name}.trk"
    python3 $quickbundles_script $trk_fn $output_trk_base $subject $tract_name $num_sl $points_per_sl
done


for trk_name in `ls -1 $output_trk_base | sed 's/\.trk$//g'`
do
    trk_fn="${output_trk_base}/${trk_name}.trk"

    # Since all HCP data shares same affine, can use any subject as the affine reference
    ref_affine_file="../../../DATASETS/HCP_100_SUBJECTS/100307/T1w/Diffusion/nodif_brain_mask.nii.gz" 

    ### GENERATE A TRACT MASK ###
    is_legacy=1 # set to 1 if using legacy data (dataset <=V1.1.0), else set to 0
    python3 $generate_masks $trk_fn ${output_masks_base}/${trk_name}.nii.gz $ref_affine_file $is_legacy

    # Generate endings (beginning/end) masks
    python3 $generate_endings $ref_affine_file $trk_fn ${output_endings_base}/${trk_name}

    # Generate a TOM
    mask="${output_masks_base}/${trk_name}.nii.gz"
    sh $generate_toms -i $trk_fn -o ${output_toms_base}/$trk_name --mask $mask --athresh 10 --peakthresh 0.1 --numdirs 1 --normalization 3 --file_ending .nii.gz

    # Pre-process the generated nifti files
    tom_in="${output_toms_base}/${trk_name}_DIRECTIONS.nii.gz"
    tom_out="${preprocessed_toms}/${trk_name}.nii.gz"

    mask_in="${output_masks_base}/${trk_name}.nii.gz"
    mask_out="${preprocessed_masks}/${trk_name}.nii.gz"

    beginning_in="${output_endings_base}/${trk_name}_beginnings.nii.gz"
    beginning_out="${preprocessed_beginnings}/${trk_name}.nii.gz"

    ending_in="${output_endings_base}/${trk_name}_endings.nii.gz"
    ending_out="${preprocessed_endings}/${trk_name}.nii.gz"

    python3 $preprocess $tom_in $tom_out $mask_in $mask_out $beginning_in $beginning_out $ending_in $ending_out
done
